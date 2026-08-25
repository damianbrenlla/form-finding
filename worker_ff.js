/**
 * DBSW 3D Form-Finding WebWorker Engine
 * Author: Damian Brenlla / DBSW 2026
 * v7 — Updated for 3D Line Supports & Dynamic Bounding Geometry
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
            let response;
            try {
                response = await fetch(url);
            } catch (networkErr) {
                throw new Error(
                    `Network error fetching ${file} at ${url}: ${networkErr.message}. ` +
                    `Serve via local server if testing locally.`
                );
            }
            if (!response.ok) {
                throw new Error(
                    `HTTP ${response.status} ${response.statusText} fetching ${file} at ${url}. ` +
                    `Check that python_core/${file} exists alongside worker_ff.js.`
                );
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
from core.solvers_ff import UniversalFormFindingSolver

payload = json.loads(payload_json)

# --- Material Properties ---
mat_props = FormFindingMaterialRegistry.resolve_properties(payload)
mat_type  = payload.get("material_type", "generic")

# --- Domain ---
Lx_val = float(payload.get("Lx", 6000))
Ly_val = float(payload.get("Ly", 300))
Lz_val = float(payload.get("Lz", 600))

domain = FormFindingDomain3D(
    Lx=Lx_val,
    Ly=Ly_val,
    Lz=Lz_val,
    nx=int(payload.get("nx", 36)),
    ny=int(payload.get("ny", 9)),
    geometry_preset="surface_grid"
)

# --- Discrete Point Supports ---
for pt in payload.get("point_supports", []):
    domain.add_point_support(
        float(pt.get("x", 0)),
        float(pt.get("y", 0)),
        float(pt.get("z", 0))
    )

# --- Discrete Line Supports ---
for l_sup in payload.get("line_supports", []):
    p1 = (float(l_sup.get("x1", 0)), float(l_sup.get("y1", 0)), float(l_sup.get("z1", 0)))
    p2 = (float(l_sup.get("x2", 0)), float(l_sup.get("y2", 0)), float(l_sup.get("z2", 0)))
    
    # Check if domain supports 3D line vector constraints
    if hasattr(domain, 'add_line_support_3d'):
        domain.add_line_support_3d(p1, p2)
    else:
        # Fallback to endpoints if 3D line method isn't declared
        domain.add_point_support(*p1)
        domain.add_point_support(*p2)

if len(domain.fixed_nodes) == 0:
    raise ValueError(
        "No support nodes resolved. Add at least one discrete point or line support."
    )

# --- Cross-Section Area ---
if mat_type == "cables":
    d_mm     = max(float(payload.get("sec_cable_d", 24.0)), 1.0)
    area_mm2 = np.pi * (d_mm / 2.0) ** 2
elif mat_type == "membrane":
    t_mm     = max(float(payload.get("sec_fabric_t", 1.2)), 0.1)
    area_mm2 = t_mm * 1000.0
else:
    b_mm     = max(float(payload.get("sec_b", 300.0)), 1.0)
    h_mm     = max(float(payload.get("sec_h", 300.0)), 1.0)
    area_mm2 = b_mm * h_mm

# --- Prestress ---
prestress_N = float(payload.get("prestress", 0.0))
if mat_type not in ("cables", "membrane"):
    prestress_N = 0.0

# --- Self-weight toggle ---
include_sw = bool(payload.get("include_self_weight", True))
gamma      = mat_props.get("gamma_kn_m3", 25.0) if include_sw else 0.0

# --- Solver Execution ---
solver = UniversalFormFindingSolver(
    domain          = domain,
    E_modulus       = mat_props.get("E", 210000.0),
    gamma_kn_m3     = gamma,
    area_mm2        = area_mm2,
    prestress_force = prestress_N,
    point_loads     = payload.get("loads", []),
    material_type   = mat_type,
)

invert_flag = mat_type in ("concrete", "masonry")
iters       = 500

initial_nodes = np.copy(domain.nodes).astype(float)

equilibrium_nodes, axial_forces, reactions, diagnostics = solver.solve_equilibrium(
    iterations  = iters,
    invert_form = invert_flag,
)

displacement_vecs = equilibrium_nodes - initial_nodes
deflections_mm = np.linalg.norm(displacement_vecs, axis=1)
u_max = float(np.max(deflections_mm)) if len(deflections_mm) > 0 else 0.0

element_stresses_mpa = axial_forces / max(area_mm2, 1e-4)

# Nodal Stress Averaging
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

# Reactions
fixed_indices = sorted(list(domain.fixed_nodes))
reaction_data = []
for idx in fixed_indices:
    pos      = [float(v) for v in np.nan_to_num(equilibrium_nodes[idx]).tolist()]
    rx, ry, rz = [float(v) for v in np.nan_to_num(reactions[idx]).tolist()]
    R_total  = float(np.linalg.norm([rx, ry, rz]))
    reaction_data.append({
        "node":       int(idx),
        "pos":        pos,
        "Rx_kN":      round(rx / 1000.0, 3),
        "Ry_kN":      round(ry / 1000.0, 3),
        "Rz_kN":      round(rz / 1000.0, 3),
        "R_total_kN": round(R_total / 1000.0, 3),
    })

# Serialization
clean_nodes    = np.nan_to_num(equilibrium_nodes, nan=0.0, posinf=0.0, neginf=0.0)
clean_forces   = np.nan_to_num(axial_forces,        nan=0.0, posinf=0.0, neginf=0.0)
clean_stresses = np.nan_to_num(nodal_stresses_mpa, nan=0.0, posinf=0.0, neginf=0.0)
clean_defs     = np.nan_to_num(deflections_mm,     nan=0.0, posinf=0.0, neginf=0.0)

json.dumps({
    "nodes":          [[float(v) for v in row] for row in clean_nodes.tolist()],
    "edges":          [[int(v)   for v in row] for row in np.asarray(domain.edges, dtype=int).tolist()],
    "axial_forces":   [float(v) for v in clean_forces.tolist()],
    "stresses_mpa":   [float(v) for v in clean_stresses.tolist()],
    "deflections_mm": [float(v) for v in clean_defs.tolist()],
    "sigma_max_tens": round(sigma_max_tens, 3),
    "sigma_max_comp": round(sigma_max_comp, 3),
    "u_max":          round(u_max, 3),
    "reactions":      reaction_data,
    "material":       mat_props.get("material_name", mat_type),
    "num_nodes":      len(clean_nodes),
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
