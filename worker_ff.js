/**
 * DBSW 3D Form-Finding WebWorker Engine (Worker FF)
 * Author: Damian Brenlla / DBSW 2026
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

async function initEngine() {
    try {
        postMessage({ status: "log", message: "Initialising Pyodide WebAssembly runtime..." });
        
        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/"
        });

        postMessage({ status: "log", message: "Loading NumPy package into Wasm memory..." });
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
        postMessage({ status: "error", message: err.toString() });
    }
}

self.onmessage = async function(e) {
    const { action, payload } = e.data;

    if (action === "init") {
        await initEngine();
        return;
    }

    if (action === "solve" || action === "form_find") {
        pyodide.globals.set("payload_json", JSON.stringify(payload));

        try {
            const resultJson = await pyodide.runPythonAsync(`
import json, numpy as np
from core.domain_ff import FormFindingDomain3D
from core.materials import FormFindingMaterialRegistry
from core.solvers_ff import UniversalFormFindingSolver

payload = json.loads(payload_json)
mat_props = FormFindingMaterialRegistry.resolve_properties(payload)

domain = FormFindingDomain3D(
    Lx=float(payload.get("Lx", 6000)), Ly=float(payload.get("Ly", 6000)), Lz=float(payload.get("Lz", 2000)),
    nx=int(payload.get("nx", 16)), ny=int(payload.get("ny", 16)),
    geometry_preset=payload.get("preset", "vault")
)

sup_preset = payload.get("support_preset", "four_corners")
if sup_preset == "four_corners":
    domain.add_point_support(0.0, 0.0, 0.0)
    domain.add_point_support(domain.Lx, 0.0, 0.0)
    domain.add_point_support(0.0, domain.Ly, 0.0)
    domain.add_point_support(domain.Lx, domain.Ly, 0.0)
elif sup_preset == "two_opposite_lines":
    domain.add_line_support("x", 0.0)
    domain.add_line_support("x", domain.Lx)

solver = UniversalFormFindingSolver(
    domain=domain, E_modulus=mat_props["E"], gamma_kn_m3=mat_props["gamma_kn_m3"],
    prestress_force=float(payload.get("prestress", 15.0))
)

invert_flag = payload.get("preset") in ["vault", "dome", "catenary_arch"]
equilibrium_nodes, axial_forces, reactions = solver.solve_equilibrium(
    iterations=int(payload.get("iterations", 300)),
    invert_form=invert_flag
)

fixed_indices = list(domain.fixed_nodes)
reaction_data = []
for idx in fixed_indices:
    pos = [float(x) for x in equilibrium_nodes[idx].tolist()]
    rx, ry, rz = [float(f) for f in reactions[idx].tolist()]
    reaction_data.append({
        "node": int(idx),
        "pos": pos,
        "Rx_kN": round(rx / 1000.0, 2),
        "Ry_kN": round(ry / 1000.0, 2),
        "Rz_kN": round(rz / 1000.0, 2),
        "R_total_kN": round(float(np.linalg.norm([rx, ry, rz])) / 1000.0, 2)
    })

# Convert all NumPy ndarrays/types to native Python primitives for JSON serialization
nodes_list = [[float(val) for val in row] for row in equilibrium_nodes.tolist()]
edges_list = [[int(val) for val in row] for row in np.asarray(domain.edges, dtype=int).tolist()]
forces_list = [float(val) for val in np.asarray(axial_forces, dtype=float).tolist()]

json.dumps({
    "nodes": nodes_list,
    "edges": edges_list,
    "axial_forces": forces_list,
    "reactions": reaction_data,
    "material": mat_props["material_name"]
})
            `);

            postMessage({ status: "completed", data: JSON.parse(resultJson) });
        } catch (err) {
            postMessage({ status: "error", message: err.toString() });
        }
    }
};
