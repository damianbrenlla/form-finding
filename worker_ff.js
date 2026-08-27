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
        
        const [domainRes, materialsRes, solversRes] = await Promise.all([
            fetch('python_core/domain_ff.py'),
            fetch('python_core/materials.py'),
            fetch('python_core/solvers_ff.py')
        ]);

        if (!domainRes.ok || !materialsRes.ok || !solversRes.ok) {
            throw new Error(`HTTP Fetch Failed: domain (${domainRes.status}), materials (${materialsRes.status}), solvers (${solversRes.status})`);
        }

        const domainCode = await domainRes.text();
        const materialsCode = await materialsRes.text();
        const solversCode = await solversRes.text();

        try {
            pyodide.FS.mkdir('/home/pyodide/python_core');
        } catch (e) {
            // Directory exists across re-inits
        }

        pyodide.FS.writeFile('/home/pyodide/python_core/__init__.py', '');
        pyodide.FS.writeFile('/home/pyodide/python_core/domain_ff.py', domainCode);
        pyodide.FS.writeFile('/home/pyodide/python_core/materials.py', materialsCode);
        pyodide.FS.writeFile('/home/pyodide/python_core/solvers_ff.py', solversCode);

        await pyodide.runPythonAsync(`
import sys
import importlib

if '/home/pyodide' not in sys.path:
    sys.path.insert(0, '/home/pyodide')

importlib.invalidate_caches()

from python_core.domain_ff import FormFindingDomain3D
from python_core.solvers_ff import FormFindingSolverFactory

assert 'FormFindingDomain3D' in globals(), "Scope injection failed: FormFindingDomain3D missing"
assert 'FormFindingSolverFactory' in globals(), "Scope injection failed: FormFindingSolverFactory missing"
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

# Smooth Nodal Cauchy Stress Averaging
nodal_forces = np.zeros(len(solved_nodes), dtype=float)
nodal_edge_counts = np.zeros(len(solved_nodes), dtype=float)

for i, (u, v) in enumerate(domain.edges):
    f_abs = abs(axial_forces[i])
    nodal_forces[u] += f_abs
    nodal_forces[v] += f_abs
    nodal_edge_counts[u] += 1.0
    nodal_edge_counts[v] += 1.0

nodal_edge_counts = np.maximum(nodal_edge_counts, 1.0)
stresses_mpa = (nodal_forces / (nodal_edge_counts * max(sec_t, 0.1))).tolist()

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
