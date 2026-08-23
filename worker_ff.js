/**
 * DBSW 3D Form-Finding WebWorker Engine
 * Author: Damian Brenlla / DBSW 2026
 * v5 — Fixed: Computes axial stress (MPa) and displacement magnitudes (mm) for field mapping.
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
                    `If you are opening this page as a file:// URL, serve it with a local ` +
                    `web server instead (e.g. "python3 -m http.server") — workers cannot ` +
                    `fetch local files over file://.`
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
        domain.add_point_support(0.0,       0.0,       0.0)
        domain.add_point_support(domain.Lx,  0.0,       0.0)
        domain.add_point_support(0.0,       domain.Ly,  0.0)
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

if len(domain.fixed_nodes) == 0:
    raise ValueError(
        "No support nodes resolved. If using 'Only User-Selected Discrete "
        "Points' mode, add at least one row to the Discrete Point Supports "
        "table, or switch back to a Support Preset."
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

preset       = payload.get("preset", "surface_grid")
invert_flag  = preset in ("vault", "dome")
iters        = int(payload.get("iterations", 500))

# Save initial node layout to measure displacement
initial_nodes = np.copy(domain.nodes).astype(float)

equilibrium_nodes, axial_forces, reactions, diagnostics = solver.solve_equilibrium(
    iterations  = iters,
    invert_form = invert_flag,
)

# Compute per-node displacement magnitudes (mm)
displacement_vecs = equilibrium_nodes - initial_nodes
deflections_mm = np.linalg.norm(displacement_vecs, axis=1)
u_max = float(np.max(deflections_mm)) if len(deflections_mm) > 0 else 0.0

# Compute per-element axial stresses sigma = N / A (MPa)
stresses_mpa = axial_forces / max(area_mm2, 1e-4)
sigma_max_tens = float(np.max(stresses_mpa)) if len(stresses_mpa) > 0 else 0.0
sigma_max_comp = float(np.min(stresses_mpa)) if len(stresses_mpa) > 0 else 0.0

# --- Reactions ---
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

# --- Clean output ---
clean_nodes  = np.nan_to_num(equilibrium_nodes, nan=0.0, posinf=0.0, neginf=0.0)
clean_forces = np.nan_to_num(axial_forces,      nan=0.0, posinf=0.0, neginf=0.0)
clean_stresses = np.nan_to_num(stresses_mpa,    nan=0.0, posinf=0.0, neginf=0.0)
clean_defs   = np.nan_to_num(deflections_mm,    nan=0.0, posinf=0.0, neginf=0.0)

nodes_list  = [[float(v) for v in row] for row in clean_nodes.tolist()]
edges_list  = [[int(v)   for v in row] for row in np.asarray(domain.edges, dtype=int).tolist()]
forces_list = [float(v) for v in clean_forces.tolist()]
stresses_list = [float(v) for v in clean_stresses.tolist()]
defs_list   = [float(v) for v in clean_defs.tolist()]

json.dumps({
    "nodes":          nodes_list,
    "edges":          edges_list,
    "axial_forces":   forces_list,
    "stresses_mpa":   stresses_list,
    "deflections_mm": defs_list,
    "sigma_max_tens": round(sigma_max_tens, 3),
    "sigma_max_comp": round(sigma_max_comp, 3),
    "u_max":          round(u_max, 3),
    "reactions":      reaction_data,
    "material":       mat_props["material_name"],
    "preset":         preset,
    "num_nodes":      len(nodes_list),
    "num_edges":      len(edges_list),
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
