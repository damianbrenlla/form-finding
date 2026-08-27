// worker_ff.js — DBSW WebAssembly Worker Bridge
importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

async function initPyodideRuntime() {
    try {
        self.postMessage({ status: 'log', message: 'Downloading Pyodide core & WebAssembly engine...' });
        pyodide = await loadPyodide();
        
        self.postMessage({ status: 'log', message: 'Compiling NumPy vectorisation library...' });
        await pyodide.loadPackage(["numpy"]);

        self.postMessage({ status: 'log', message: 'Fetching DBSW core domain and solver logic from /python_core...' });
        
        // 1. Fetch Python source files from the python_core/ directory
        const [domainRes, materialsRes, solversRes] = await Promise.all([
            fetch('python_core/domain_ff.py'),
            fetch('python_core/materials.py'),
            fetch('python_core/solvers_ff.py')
        ]);

        if (!domainRes.ok || !materialsRes.ok || !solversRes.ok) {
            throw new Error("Failed to fetch one or more Python files from 'python_core/'. Check server paths.");
        }

        const domainCode = await domainRes.text();
        const materialsCode = await materialsRes.text();
        const solversCode = await solversRes.text();

        // 2. Write them directly to Pyodide's in-memory Virtual File System (FS)
        // This ensures Python standard imports like `import python_core.domain_ff` or `from domain_ff import *` work properly.
        pyodide.FS.mkdir('/home/pyodide/python_core');
        pyodide.FS.writeFile('/home/pyodide/python_core/__init__.py', '');
        pyodide.FS.writeFile('/home/pyodide/python_core/domain_ff.py', domainCode);
        pyodide.FS.writeFile('/home/pyodide/python_core/materials.py', materialsCode);
        pyodide.FS.writeFile('/home/pyodide/python_core/solvers_ff.py', solversCode);

        // 3. Inject modules into Pyodide's global execution scope
        await pyodide.runPythonAsync(`
import sys
sys.path.append('/home/pyodide')

from python_core.domain_ff import FormFindingDomain3D
from python_core.solvers_ff import FormFindingSolverFactory
`);

        self.postMessage({ status: 'ready' });
    } catch (err) {
        self.postMessage({ status: 'error', message: "Initialization error: " + err.message });
    }
}

self.onmessage = async function(e) {
    const { action, payload } = e.data;

    if (action === 'init') {
        await initPyodideRuntime();
    } else if (action === 'form_find') {
        if (!pyodide) {
            self.postMessage({ status: 'error', message: 'Pyodide engine is not initialized yet.' });
            return;
        }

        try {
            self.postMessage({ status: 'log', message: 'Executing FE Equilibrium Solver...' });
            
            pyodide.globals.set("js_payload", JSON.stringify(payload));

            const runnerScript = `
import json
import numpy as np
from python_core.domain_ff import FormFindingDomain3D
from python_core.solvers_ff import FormFindingSolverFactory

payload = json.loads(js_payload)

mat_type = str(payload.get("material_type", "membrane")).lower()
nx = int(payload.get("nx", 36))
ny = int(payload.get("ny", 12))

point_supports = payload.get("point_supports", [])
line_supports = payload.get("line_supports", [])

perim_pts = []
inter_pts = []

for ps in point_supports:
    role = str(ps.get("role", "perimeter")).lower()
    pt = [float(ps["x"]), float(ps["y"]), float(ps["z"])]
    if role in ("perimeter", "external", "boundary"):
        perim_pts.append(pt)
    else:
        inter_pts.append(pt)

ext_lines = []
int_lines = []
for ls in line_supports:
    role = str(ls.get("role", "external")).lower()
    seg = [[float(ls["x1"]), float(ls["y1"]), float(ls["z1"])], [float(ls["x2"]), float(ls["y2"]), float(ls["z2"])]]
    if role in ("external", "boundary", "perimeter"):
        ext_lines.append(seg)
    else:
        int_lines.append(seg)

# Domain Construction Logic
if mat_type in ("cables", "cable"):
    domain = FormFindingDomain3D(
        xmin=0.0, xmax=float(payload.get("Lx", 6000)),
        ymin=0.0, ymax=float(payload.get("Ly", 3000)),
        Lz=float(payload.get("Lz", 1000)),
        nx=nx, ny=1, material_type="cables"
    )
    for ps in point_supports:
        domain.add_point_support(float(ps["x"]), float(ps["y"]), float(ps["z"]))
elif len(ext_lines) >= 3:
    domain = FormFindingDomain3D.build_polygon_domain_from_line_segments(
        boundary_segments=ext_lines, interior_points=inter_pts, interior_line_segments=int_lines,
        nx=nx, ny=ny, Lz=float(payload.get("Lz", 1000)), material_type="membrane"
    )
elif len(perim_pts) >= 3:
    domain = FormFindingDomain3D.build_polygon_domain(
        perimeter_points=perim_pts, interior_points=inter_pts, interior_line_segments=int_lines,
        nx=nx, ny=ny, Lz=float(payload.get("Lz", 1000)), material_type="membrane"
    )
else:
    domain = FormFindingDomain3D(
        xmin=0.0, xmax=float(payload.get("Lx", 6000)),
        ymin=0.0, ymax=float(payload.get("Ly", 3000)),
        Lz=float(payload.get("Lz", 1000)),
        nx=nx, ny=ny, material_type="membrane"
    )
    domain.add_edge_support("all")

# Section & Solver Selection
sec_d = float(payload.get("sec_cable_d", 24.0))
sec_t = float(payload.get("sec_fabric_t", 1.2))
area = (np.pi * (sec_d**2) / 4.0) if mat_type in ("cables", "cable") else (sec_t * 1.0)

mat_props = {
    "E": float(payload.get("custom_E", 210000.0)),
    "gamma": float(payload.get("custom_gamma_kn_m3", 78.5))
}

solver = FormFindingSolverFactory.create(
    material_type=mat_type, domain=domain, mat_props=mat_props,
    area_mm2=area, prestress_force=float(payload.get("prestress", 0.0)),
    prestress_warp_N_mm=float(payload.get("prestress_warp_kn_m", 2.0)),
    prestress_weft_N_mm=float(payload.get("prestress_weft_kn_m", 2.0)),
    edge_cable_prestress_N=float(payload.get("edge_cable_prestress_kn", 20.0)) * 1000.0,
    point_loads=payload.get("loads", []),
    gamma_kn_m3=mat_props["gamma"] if payload.get("include_self_weight", True) else 0.0
)

solved_nodes, axial_forces, reactions, diagnostics = solver.solve_equilibrium(iterations=1200)

stresses_mpa = (axial_forces / max(area, 1e-4)).tolist()
initial_nodes = domain.nodes.astype(float)
deflections_mm = np.linalg.norm(solved_nodes - initial_nodes, axis=1).tolist()

reaction_list = []
for idx in sorted(list(domain.fixed_nodes)):
    rx, ry, rz = reactions[idx]
    r_tot = float(np.linalg.norm(reactions[idx])) / 1000.0
    reaction_list.append({
        "node_index": int(idx),
        "pos": solved_nodes[idx].tolist(),
        "Rx_kN": float(rx) / 1000.0,
        "Ry_kN": float(ry) / 1000.0,
        "Rz_kN": float(rz) / 1000.0,
        "R_total_kN": r_tot
    })

result = {
    "nodes": solved_nodes.tolist(),
    "edges": domain.edges.tolist(),
    "triangles": domain.triangles.tolist(),
    "axial_forces": axial_forces.tolist(),
    "stresses_mpa": stresses_mpa,
    "deflections_mm": deflections_mm,
    "reactions": reaction_list,
    "diagnostics": diagnostics,
    "material": mat_type,
    "nx_actual": int(domain.nx),
    "ny_actual": int(domain.ny)
}

json.dumps(result)
`;

            let jsonResult = await pyodide.runPythonAsync(runnerScript);
            self.postMessage({ status: 'completed', data: JSON.parse(jsonResult) });

        } catch (err) {
            self.postMessage({ status: 'error', message: err.message });
        }
    }
};
