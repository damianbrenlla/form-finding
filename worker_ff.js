/**
 * DBSW 3D Form-Finding WebWorker Engine
 * Author: Damian Brenlla / DBSW 2026
 * v22 — Loads SciPy alongside NumPy for constrained polygon Delaunay triangulation,
 *        routes perimeter/interior support points directly to FormFindingDomain3D.build_polygon_domain(),
 *        and passes explicit face triangles to the frontend rendering pipeline.
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
        postMessage({ status: "log", message: "Loading NumPy & SciPy into Wasm memory..." });
        await pyodide.loadPackage(["numpy", "scipy"]);
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

mat_type = str(payload.get("material_type") or payload.get("material_grade") or "cables").lower()
if "cable" in mat_type or "rope" in mat_type or "strand" in mat_type:
    payload["material_type"] = "cables"
elif "fabric" in mat_type or "ptfe" in mat_type or "membrane" in mat_type:
    payload["material_type"] = "membrane"
elif "concrete" in mat_type or "c20" in mat_type or "c30" in mat_type:
    payload["material_type"] = "concrete"

mat_props = FormFindingMaterialRegistry.resolve_properties(payload)
mat_type  = mat_props.get("material_type", "cables")

Lx_val = float(payload.get("Lx", 6000))
Ly_val = float(payload.get("Ly", 3000))
Lz_val = float(payload.get("Lz", 1000))
point_sups = payload.get("point_supports", [])
line_sups  = payload.get("line_supports", [])

# Separate Point Supports by Role
perimeter_sups = [pt for pt in point_sups if pt.get("role", "perimeter") == "perimeter"]
interior_sups  = [pt for pt in point_sups if pt.get("role", "") == "interior"]

target_edge_len = float(payload.get("target_edge_len", 250.0))

# ------------------------------------------------------------------
# POLYGON DOMAIN ROUTING
# Triggered for membranes whenever >=3 perimeter points exist
# ------------------------------------------------------------------
if mat_type in ("membrane", "fabric") and len(perimeter_sups) >= 3:
    domain = FormFindingDomain3D.build_polygon_domain(
        perimeter_pts=perimeter_sups,
        interior_pts=interior_sups,
        line_supports=line_sups,
        target_edge_len=target_edge_len,
        material_type=mat_type
    )
# ------------------------------------------------------------------
# CABLE 1D POLYLINE ROUTING
# ------------------------------------------------------------------
elif mat_type in ("cables", "cable") and len(point_sups) >= 2:
    pA = point_sups[0]
    pB = point_sups[1]
    x1, y1, z1 = float(pA.get("x",0)), float(pA.get("y",0)), float(pA.get("z",0))
    x2, y2, z2 = float(pB.get("x",0)), float(pB.get("y",0)), float(pB.get("z",0))
    nx = max(int(payload.get("nx", 36)), 2)

    nodes = np.zeros((nx + 1, 3), dtype=float)
    for i in range(nx + 1):
        t = i / nx
        nodes[i, 0] = x1 + t * (x2 - x1)
        nodes[i, 1] = y1 + t * (y2 - y1)
        nodes[i, 2] = z1 + t * (z2 - z1)

    edges = np.array([[i, i + 1] for i in range(nx)], dtype=int)
    faces = np.empty((0, 3), dtype=int)
    domain = FormFindingDomain3D(
        nodes=nodes,
        edges=edges,
        faces=faces,
        fixed_nodes={0, nx},
        perimeter_nodes={0, nx},
        material_type=mat_type,
        Lx=abs(x2-x1), Ly=abs(y2-y1), Lz=Lz_val
    )
else:
    raise ValueError("Invalid boundary configuration: Minimum 3 ordered perimeter points required for irregular membranes.")

# Seed interior elevations for initial surface curvature
seed_z_map = {}
for pt in point_sups:
    seed_z_map[(float(pt.get("x", 0)), float(pt.get("y", 0)))] = float(pt.get("z", 0))

if mat_type not in ("cables", "cable"):
    domain.apply_idw_surface_interpolation(seed_z_map)

# Prestress & Material Geometry Resolution
prestress_warp_N_mm = 0.0
prestress_weft_N_mm = 0.0
edge_cable_prestress_N = 0.0
prestress_N = 0.0

if mat_type in ("cables", "cable"):
    d_mm     = max(float(payload.get("sec_cable_d", 24.0)), 1.0)
    area_mm2 = np.pi * (d_mm / 2.0) ** 2
    prestress_N = float(payload.get("prestress", 0.0))
else:
    t_mm     = max(float(payload.get("sec_fabric_t", 1.2)), 0.1)
    area_mm2 = t_mm * 1000.0
    prestress_warp_N_mm = float(payload.get("prestress_warp_kn_m", 2.0))
    prestress_weft_N_mm = float(payload.get("prestress_weft_kn_m", 2.0))
    edge_cable_prestress_N = float(payload.get("edge_cable_prestress_kn", 20.0)) * 1000.0

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
    "faces":          [[int(v)   for v in row] for row in np.asarray(domain.faces, dtype=int).tolist()],
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
