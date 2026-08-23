/**
 * DBSW 3D Form-Finding WebWorker Engine
 * Author: Damian Brenlla / DBSW 2026
 * v2 — Fixed: material_type passed to solver, invert logic, payload mapping
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

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
            const response = await fetch(`./python_core/${file}?cb=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status} fetching python_core/${file}`);
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
domain = FormFindingDomain3D(
    Lx=float(payload.get("Lx", 6000)),
    Ly=float(payload.get("Ly", 6000)),
    Lz=float(payload.get("Lz", 2000)),
    nx=int(payload.get("nx", 16)),
    ny=int(payload.get("ny", 16)),
    geometry_preset=payload.get("preset", "surface_grid"),
)

# --- Boundary Restraints ---
sup_mode   = payload.get("support_mode", "preset")
sup_preset = payload.get("support_preset", "four_corners")

if sup_mode == "points_only" and payload.get("point_supports"):
    for pt in payload["point_supports"]:
        domain.add_point_support(
            float(pt.get("x", 0)),
            float(pt.get("y", 0)),
            float(pt.get("z", 0))
        )
else:
    if sup_preset == "four_corners":
        domain.add_point_support(0.0,        0.0,        0.0)
        domain.add_point_support(domain.Lx,  0.0,        0.0)
        domain.add_point_support(0.0,        domain.Ly,  0.0)
        domain.add_point_support(domain.Lx,  domain.Ly,  0.0)
    elif sup_preset == "two_opposite_lines":
        domain.add_line_support("x", 0.0)
        domain.add_line_support("x", domain.Lx)

# Also apply any additional discrete point supports from the table
for pt in payload.get("point_supports", []):
    domain.add_point_support(
        float(pt.get("x", 0)),
        float(pt.get("y", 0)),
        float(pt.get("z", 0))
    )

# --- Cross-Section Area ---
if mat_type == "cable":
    d_mm     = max(float(payload.get("sec_cable_d", 24.0)), 1.0)
    area_mm2 = np.pi * (d_mm / 2.0) ** 2
elif mat_type == "fabric":
    t_mm     = max(float(payload.get("sec_fabric_t", 1.2)), 0.1)
    area_mm2 = t_mm * 1000.0          # per metre width strip
else:
    b_mm     = max(float(payload.get("sec_b", 300.0)), 1.0)
    h_mm     = max(float(payload.get("sec_h", 300.0)), 1.0)
    area_mm2 = b_mm * h_mm

# --- Prestress ---
# Only meaningful for cables and fabrics
prestress_N = float(payload.get("prestress", 0.0))
if mat_type not in ("cable", "fabric"):
    prestress_N = 0.0

# --- Self-weight toggle ---
include_sw = bool(payload.get("include_self_weight", True))
gamma      = mat_props["gamma_kn_m3"] if include_sw else 0.0

# --- Solver ---
solver = UniversalFormFindingSolver(
    domain          = domain,
    E_modulus       = mat_props["E"],
    gamma_kn_m3     = gamma,
    area_mm2        = area_mm2,
    prestress_force = prestress_N,
    point_loads     = payload.get("loads", []),
    material_type   = mat_type,
)

# Invert form for compression presets (vault, dome, catenary_arch)
preset       = payload.get("preset", "surface_grid")
invert_flag  = preset in ("vault", "dome")

# Increase iterations for complex geometries
iters = int(payload.get("iterations", 500))

equilibrium_nodes, axial_forces, reactions = solver.solve_equilibrium(
    iterations  = iters,
    invert_form = invert_flag,
)

# --- Reactions ---
fixed_indices = sorted(list(domain.fixed_nodes))
reaction_data = []
for idx in fixed_indices:
    pos      = [float(v) for v in np.nan_to_num(equilibrium_nodes[idx]).tolist()]
    rx, ry, rz = [float(v) for v in np.nan_to_num(reactions[idx]).tolist()]
    R_total  = float(np.linalg.norm([rx, ry, rz]))
    reaction_data.append({
        "node":      int(idx),
        "pos":       pos,
        "Rx_kN":     round(rx / 1000.0, 3),
        "Ry_kN":     round(ry / 1000.0, 3),
        "Rz_kN":     round(rz / 1000.0, 3),
        "R_total_kN": round(R_total / 1000.0, 3),
    })

# --- Clean output ---
clean_nodes  = np.nan_to_num(equilibrium_nodes, nan=0.0, posinf=0.0, neginf=0.0)
clean_forces = np.nan_to_num(axial_forces,      nan=0.0, posinf=0.0, neginf=0.0)

nodes_list  = [[float(v) for v in row] for row in clean_nodes.tolist()]
edges_list  = [[int(v)   for v in row] for row in np.asarray(domain.edges, dtype=int).tolist()]
forces_list = [float(v) for v in clean_forces.tolist()]

json.dumps({
    "nodes":       nodes_list,
    "edges":       edges_list,
    "axial_forces": forces_list,
    "reactions":   reaction_data,
    "material":    mat_props["material_name"],
    "preset":      preset,
    "num_nodes":   len(nodes_list),
    "num_edges":   len(edges_list),
})
`);

            postMessage({ status: "completed", data: JSON.parse(resultJson) });

        } catch (err) {
            postMessage({ status: "error", message: err.toString() });
        }
    }
};
