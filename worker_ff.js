/**
 * DBSW 3D Form-Finding WebWorker Engine
 * Author: Damian Brenlla / DBSW 2026
 * v19 — Origin-aware domain: true bounding box computed from ALL support
 *       coordinates (including negative ones), every support coordinate forced
 *       onto an exact grid line so it can never fail to attach, and IDW seeding
 *       across every support instead of a 4-corner-only bilinear bucket.
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

function corePythonUrl(filename) {
    return new URL(`./python_core/${filename}`, self.location.href).href;
}

async function initEngine() {
    try {
        postMessage({ status: "log", message: "Initialising Pyodide WebAssembly runtime..." });

        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/"
        });

        postMessage({ status: "log", message: "Loading NumPy into Wasm memory..." });
        await pyodide.loadPackage("numpy");

        postMessage({ status: "log", message: "Mounting Python core files..." });
        pyodide.FS.mkdirTree("/home/pyodide/core");

        const files = ["domain_ff.py", "materials.py", "solvers_ff.py"];
        for (const file of files) {
            const url = corePythonUrl(file) + `?cb=${Date.now()}`;
            let response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status} fetching ${file} at ${url}.`);
            }
            const code = await response.text();
            pyodide.FS.writeFile(`/home/pyodide/core/${file}`, code);
        }

        await pyodide.runPythonAsync(`
import sys
if '/home/pyodide' not in sys.path:
    sys.path.append('/home/pyodide')
`);

        postMessage({ status: "ready" });

    } catch (err) {
        console.error("[worker_ff.js] Init failed:", err);
        postMessage({ status: "error", message: "Init failed: " + err.toString() });
    }
}

self.onmessage = async function(e) {
    const { action, payload } = e.data;

    if (action === "init") {
        await initEngine();
        return;
    }

    if (action === "solve" || action === "form_find") {
        if (!pyodide) {
            postMessage({ status: "error", message: "Engine not initialised." });
            return;
        }

        pyodide.globals.set("payload_json", JSON.stringify(payload));

        try {
            const resultJson = await pyodide.runPythonAsync(`
import json
import numpy as np
from core.domain_ff import FormFindingDomain3D
from core.materials import FormFindingMaterialRegistry
from core.solvers_ff import FormFindingSolverFactory

payload = json.loads(payload_json)

# Fallback Material Type Resolution
mat_type = str(payload.get("material_type") or payload.get("material_grade") or "cables").lower()
if "cable" in mat_type or "rope" in mat_type or "strand" in mat_type:
    payload["material_type"] = "cables"
elif "fabric" in mat_type or "ptfe" in mat_type or "membrane" in mat_type:
    payload["material_type"] = "membrane"
elif "concrete" in mat_type or "c20" in mat_type or "c30" in mat_type:
    payload["material_type"] = "concrete"
elif "timber" in mat_type or "gl" in mat_type or "c24" in mat_type:
    payload["material_type"] = "timber"
elif "masonry" in mat_type or "mortar" in mat_type or "brick" in mat_type:
    payload["material_type"] = "masonry"

mat_props = FormFindingMaterialRegistry.resolve_properties(payload)
mat_type  = mat_props.get("material_type", "cables")

Lx_val = float(payload.get("Lx", 6000))
Ly_val = float(payload.get("Ly", 3000))
Lz_val = float(payload.get("Lz", 1000))

point_sups = payload.get("point_supports", [])
line_sups  = payload.get("line_supports", [])
sup_preset = payload.get("support_preset", "")

# --- CRITICAL FIX: compute the TRUE bounding box from every declared support,
# including negative coordinates, instead of hard-assuming the domain starts
# at (0, 0) and spans [0, Lx] x [0, Ly]. The nominal Lx/Ly still seed the box
# so a payload with no explicit supports keeps its old default behaviour. ---
xs = [0.0, Lx_val]
ys = [0.0, Ly_val]
for pt in point_sups:
    xs.append(float(pt.get("x", 0)))
    ys.append(float(pt.get("y", 0)))
for l_sup in line_sups:
    xs.append(float(l_sup.get("x1", 0)))
    xs.append(float(l_sup.get("x2", 0)))
    ys.append(float(l_sup.get("y1", 0)))
    ys.append(float(l_sup.get("y2", 0)))

xmin_val, xmax_val = min(xs), max(xs)
ymin_val, ymax_val = min(ys), max(ys)

# --- CRITICAL FIX: force a real grid line at every support's exact X/Y so a
# support coordinate can never fail to land on a mesh node, no matter where it
# sits relative to the nominal Lx/Ly resolution. ---
forced_x = [float(pt.get("x", 0)) for pt in point_sups]
forced_y = [float(pt.get("y", 0)) for pt in point_sups]
for l_sup in line_sups:
    forced_x.append(float(l_sup.get("x1", 0)))
    forced_x.append(float(l_sup.get("x2", 0)))
    forced_y.append(float(l_sup.get("y1", 0)))
    forced_y.append(float(l_sup.get("y2", 0)))

domain = FormFindingDomain3D(
    xmin=xmin_val,
    xmax=xmax_val,
    ymin=ymin_val,
    ymax=ymax_val,
    Lz=Lz_val,
    nx=int(payload.get("nx", 36)),
    ny=int(payload.get("ny", 12)),
    forced_x=forced_x,
    forced_y=forced_y,
    geometry_preset="surface_grid",
    material_type=mat_type
)

# Discrete Point Supports & Universal Seed Z-Map Assembly
seed_z_map = {}

# Point supports processed FIRST — they have authority over corner/edge elevations.
# With forced grid lines in place, this now always finds an exact-match node;
# if it somehow doesn't, add_point_support raises instead of silently dropping
# the support (which is what was causing the fabric to drop at that corner).
for pt in point_sups:
    px, py, pz = float(pt.get("x", 0)), float(pt.get("y", 0)), float(pt.get("z", 0))
    domain.add_point_support(px, py, pz)
    seed_z_map[(px, py)] = pz

# Discrete 3D Line Supports — add geometry but do NOT overwrite a seed Z already
# set by a point support at the same coordinate.
for l_sup in line_sups:
    p1 = (float(l_sup.get("x1", 0)), float(l_sup.get("y1", 0)), float(l_sup.get("z1", 0)))
    p2 = (float(l_sup.get("x2", 0)), float(l_sup.get("y2", 0)), float(l_sup.get("z2", 0)))
    if hasattr(domain, 'add_line_support_3d'):
        domain.add_line_support_3d(p1, p2)
    if (p1[0], p1[1]) not in seed_z_map:
        seed_z_map[(p1[0], p1[1])] = p1[2]
    if (p2[0], p2[1]) not in seed_z_map:
        seed_z_map[(p2[0], p2[1])] = p2[2]

# Fallback Preset Support Resolution — now anchored to the domain's actual
# bounding box (xmin/xmax/ymin/ymax) rather than an assumed (0, Lx) span, so
# it still lands correctly even when explicit supports pushed the box negative.
if len(domain.fixed_nodes) == 0 or sup_preset == "four_corners":
    domain.add_point_support(domain.xmin, domain.ymin, 0.0)
    domain.add_point_support(domain.xmax, domain.ymin, 0.0)
    domain.add_point_support(domain.xmin, domain.ymax, 0.0)
    domain.add_point_support(domain.xmax, domain.ymax, 0.0)
elif sup_preset == "two_opposite_lines":
    if hasattr(domain, 'add_line_support_3d'):
        domain.add_line_support_3d((domain.xmin, domain.ymin, 0.0), (domain.xmin, domain.ymax, 0.0))
        domain.add_line_support_3d((domain.xmax, domain.ymin, 0.0), (domain.xmax, domain.ymax, 0.0))

if len(domain.fixed_nodes) == 0:
    raise ValueError("No support nodes resolved. Add at least one point or line support.")

# --- SEED INTERIOR 3D ELEVATIONS VIA INVERSE-DISTANCE WEIGHTING ---
# Replaces the old 4-corner-only bilinear bucket, which silently defaulted any
# support outside its four 25%-corner zones to Z=0 — that mismatch between a
# genuinely-fixed support and its Z=0 neighbours is what produced the spike.
if hasattr(domain, 'apply_idw_surface_interpolation') and mat_type not in ("cables", "cable"):
    domain.apply_idw_surface_interpolation(seed_z_map)

# --- Material Cross-Section Area & Prestress Parsing ---
prestress_warp_N_mm = 0.0
prestress_weft_N_mm = 0.0
edge_cable_prestress_N = 0.0
prestress_N = 0.0

if mat_type in ("cables", "cable"):
    d_mm     = max(float(payload.get("sec_cable_d", 24.0)), 1.0)
    area_mm2 = np.pi * (d_mm / 2.0) ** 2
    prestress_N = float(payload.get("prestress", 0.0))

elif mat_type in ("membrane", "fabric"):
    t_mm     = max(float(payload.get("sec_fabric_t", 1.2)), 0.1)
    area_mm2 = t_mm * 1000.0  # Equivalent 1m strip
    # 1 kN/m == 1 N/mm numerically; payload arrives in kN/m
    prestress_warp_N_mm = float(payload.get("prestress_warp_kn_m", 2.0))
    prestress_weft_N_mm = float(payload.get("prestress_weft_kn_m", 2.0))
    edge_cable_prestress_N = float(payload.get("edge_cable_prestress_kn", 20.0)) * 1000.0

elif mat_type == "concrete":
    t_mm     = max(float(payload.get("sec_concrete_t", 150.0)), 10.0)
    area_mm2 = t_mm * 1000.0

elif mat_type == "timber":
    b_mm     = max(float(payload.get("sec_b", 60.0)), 1.0)
    h_mm     = max(float(payload.get("sec_h", 120.0)), 1.0)
    area_mm2 = b_mm * h_mm

else:
    b_mm     = max(float(payload.get("sec_b", 300.0)), 1.0)
    h_mm     = max(float(payload.get("sec_h", 300.0)), 1.0)
    area_mm2 = b_mm * h_mm

include_sw = bool(payload.get("include_self_weight", True))
gamma      = mat_props.get("gamma_kn_m3", 25.0) if include_sw else 0.0

solver = FormFindingSolverFactory.create(
    material_type          = mat_type,
    domain                 = domain,
    mat_props              = mat_props,
    gamma_kn_m3            = gamma,
    area_mm2               = area_mm2,
    prestress_force        = prestress_N,
    prestress_warp_N_mm    = prestress_warp_N_mm,
    prestress_weft_N_mm    = prestress_weft_N_mm,
    edge_cable_prestress_N = edge_cable_prestress_N,
    point_loads            = payload.get("loads", [])
)

equilibrium_nodes, axial_forces, reactions, diagnostics = solver.solve_equilibrium(
    iterations = 1000,
    rel_tol    = 1e-4
)

displacement_vecs = equilibrium_nodes - np.copy(domain.nodes).astype(float)
deflections_mm = np.linalg.norm(displacement_vecs, axis=1)
u_max = float(np.max(deflections_mm)) if len(deflections_mm) > 0 else 0.0

element_stresses_mpa = axial_forces / max(area_mm2, 1e-4)

num_nodes = len(equilibrium_nodes)
nodal_stresses_mpa = np.zeros(num_nodes, dtype=float)
node_degree = np.zeros(num_nodes, dtype=float)

for i, (u, v) in enumerate(domain.edges):
    s = element_stresses_mpa[i]
    nodal_stresses_mpa[u] += s
    nodal_stresses_mpa[v] += s
    node_degree[u] += 1.0
    node_degree[v] += 1.0

node_degree = np.maximum(node_degree, 1.0)
nodal_stresses_mpa /= node_degree

sigma_max_tens = float(np.max(nodal_stresses_mpa)) if len(nodal_stresses_mpa) > 0 else 0.0
sigma_max_comp = float(np.min(nodal_stresses_mpa)) if len(nodal_stresses_mpa) > 0 else 0.0

fixed_indices = sorted(list(domain.fixed_nodes))
reaction_data = []
for idx in fixed_indices:
    pos        = [float(v) for v in equilibrium_nodes[idx].tolist()]
    rx, ry, rz = [float(v) for v in reactions[idx].tolist()]
    R_total    = float(np.linalg.norm([rx, ry, rz]))
    reaction_data.append({
        "node":       int(idx),
        "pos":        pos,
        "Rx_kN":      round(rx / 1000.0, 3),
        "Ry_kN":      round(ry / 1000.0, 3),
        "Rz_kN":      round(rz / 1000.0, 3),
        "R_total_kN": round(R_total / 1000.0, 3),
    })

json.dumps({
    "nodes":          [[float(v) for v in row] for row in equilibrium_nodes.tolist()],
    "edges":          [[int(v)   for v in row] for row in np.asarray(domain.edges, dtype=int).tolist()],
    "axial_forces":   [float(v) for v in axial_forces.tolist()],
    "stresses_mpa":   [float(v) for v in nodal_stresses_mpa.tolist()],
    "deflections_mm": [float(v) for v in deflections_mm.tolist()],
    "sigma_max_tens": round(sigma_max_tens, 3),
    "sigma_max_comp": round(sigma_max_comp, 3),
    "u_max":          round(u_max, 3),
    "reactions":      reaction_data,
    "material":       mat_props.get("material_name", mat_type),
    "num_nodes":      len(equilibrium_nodes),
    "num_edges":      len(domain.edges),
    "diagnostics":    diagnostics,
})
`);

            postMessage({ status: "completed", data: JSON.parse(resultJson) });

        } catch (err) {
            console.error("[worker_ff.js] Solve failed:", err);
            postMessage({ status: "error", message: err.toString() });
        }
    }
};
