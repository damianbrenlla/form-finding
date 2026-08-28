/**
 * DBSW 3D Form-Finding WebWorker Engine
 * Author: Damian Brenlla / DBSW 2026
 * v23 — Unified External/Internal point + line supports:
 *       - Polygon mode applies payload.line_supports (external lines) via
 *         domain.add_line_support_3d after the Delaunay mesh is built.
 *       - Internal line supports are sampled client-side into internal_supports
 *         (points); this worker still accepts them as fixed nodes.
 *       - Rectangular mode unchanged: point_supports + line_supports.
 *
 * v22 — Polygon (irregular) domain mode:
 *       External perimeter supports define a simple closed polygon (CW or CCW).
 *       Delaunay triangulation + strict polygon filtering builds the mesh.
 *       Internal supports are fixed inside the polygon.
 *       Returns triangles[] for mesh rendering alongside edges[] for solver.
 *       Rectangular mode (domain_mode="rectangular" or absent) is unchanged.
 *
 * v21 — Reports actual solved grid resolution (nx_actual, ny_actual)
 * v20 — Origin-aware domain + forced grid lines
 */
importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");
importScripts("./vendor/delaunator.min.js");
let pyodide = null;
function corePythonUrl(filename) {
    return new URL(`./python_core/${filename}`, self.location.href).href;
}
// ---------------------------------------------------------------------------
// POLYGON GEOMETRY HELPERS
// ---------------------------------------------------------------------------
/** Signed area (positive = CCW). */
function polygonSignedArea(pts) {
    let a = 0;
    for (let i = 0, n = pts.length; i < n; i++) {
        const j = (i + 1) % n;
        a += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
    }
    return 0.5 * a;
}
/** Point-in-polygon via ray casting. */
function pointInPolygon(px, py, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0], yi = poly[i][1];
        const xj = poly[j][0], yj = poly[j][1];
        const intersect = ((yi > py) !== (yj > py)) &&
                          (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}
/**
 * Is the polygon "simple" (no two edges crossing at interior points)?
 */
function polygonIsSimple(poly) {
    const n = poly.length;
    if (n < 4) return true;
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        for (let k = i + 1; k < n; k++) {
            const l = (k + 1) % n;
            if (i === k || i === l || j === k || j === l) continue;
            if (segmentsIntersect(
                    poly[i][0], poly[i][1], poly[j][0], poly[j][1],
                    poly[k][0], poly[k][1], poly[l][0], poly[l][1])) {
                return false;
            }
        }
    }
    return true;
}
/** Segment AB vs segment CD intersection (ignoring shared endpoints). */
function segmentsIntersect(ax, ay, bx, by, cx, cy, dx, dy) {
    const dxAB = bx - ax, dyAB = by - ay;
    const dxCD = dx - cx, dyCD = dy - cy;
    const denom = dxAB * dyCD - dyAB * dxCD;
    if (Math.abs(denom) < 1e-10) return false;
    const t = ((cx - ax) * dyCD - (cy - ay) * dxCD) / denom;
    const u = ((cx - ax) * dyAB - (cy - ay) * dxAB) / denom;
    return t > 1e-8 && t < 1 - 1e-8 && u > 1e-8 && u < 1 - 1e-8;
}
/** Shortest distance from point P to segment AB. */
function distPointToSegment(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) return Math.hypot(px - ax, py - ay);
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
/** True if point lies within `tol` of any polygon boundary edge. */
function pointOnPolygonBoundary(px, py, poly, tol) {
    const n = poly.length;
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        if (distPointToSegment(px, py, poly[i][0], poly[i][1], poly[j][0], poly[j][1]) <= tol) return true;
    }
    return false;
}
/** Inclusive point-in-polygon: strictly inside OR on the boundary (within tol). */
function pointInPolygonInclusive(px, py, poly, tol) {
    return pointInPolygon(px, py, poly) || pointOnPolygonBoundary(px, py, poly, tol);
}
/**
 * Triangle validity check for Delaunay-of-perimeter meshes.
 */
function triangleStrictlyInsidePolygon(ax, ay, bx, by, cx, cy, poly, tol) {
    tol = (tol === undefined) ? 1e-6 : tol;
    const mx = (ax + bx + cx) / 3, my = (ay + by + cy) / 3;
    if (!pointInPolygon(mx, my, poly)) return false;
    const checks = [
        [ax, ay], [bx, by], [cx, cy],
        [(ax + bx) / 2, (ay + by) / 2],
        [(bx + cx) / 2, (by + cy) / 2],
        [(ax + cx) / 2, (ay + cy) / 2]
    ];
    for (const [ex, ey] of checks) {
        if (!pointInPolygonInclusive(ex, ey, poly, tol)) return false;
    }
    const triSegs = [[ax, ay, bx, by], [bx, by, cx, cy], [ax, ay, cx, cy]];
    const n = poly.length;
    for (const [tAx, tAy, tBx, tBy] of triSegs) {
        for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            if (segmentsIntersect(tAx, tAy, tBx, tBy,
                                  poly[i][0], poly[i][1],
                                  poly[j][0], poly[j][1])) {
                return false;
            }
        }
    }
    return true;
}
/**
 * Build a Delaunay mesh over the polygon defined by perimeterPts (ordered
 * XY pairs), with holes added for strictly interior refinement grid points
 * and internalSupportPts.
 *
 * Returns { nodes, edges, triangles, fixedIndices, xmin, xmax, ymin, ymax, Lx, Ly, dx, dy }.
 */
function buildPolygonMesh(perimeterPts, internalSupportPts, nx, ny, Lz) {
    const xs = perimeterPts.map(p => p[0]);
    const ys = perimeterPts.map(p => p[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const Lx = xmax - xmin, Ly = ymax - ymin;
    const refinePts = [];
    const nxi = Math.max(nx - 1, 1), nyi = Math.max(ny - 1, 1);
    for (let i = 1; i < nxi; i++) {
        for (let j = 1; j < nyi; j++) {
            const px = xmin + (i / nxi) * Lx;
            const py = ymin + (j / nyi) * Ly;
            if (pointInPolygon(px, py, perimeterPts)) {
                refinePts.push([px, py]);
            }
        }
    }
    const allPts = [
        ...perimeterPts.map(p => [p[0], p[1]]),
        ...internalSupportPts.map(p => [p[0], p[1]]),
        ...refinePts
    ];
    const nPerim = perimeterPts.length;
    const nInternal = internalSupportPts.length;
    const coords = [];
    for (const p of allPts) { coords.push(p[0], p[1]); }
    const delaunay = new Delaunator(coords);
    const validTris = [];
    for (let t = 0; t < delaunay.triangles.length; t += 3) {
        const a = delaunay.triangles[t];
        const b = delaunay.triangles[t + 1];
        const c = delaunay.triangles[t + 2];
        const ax = allPts[a][0], ay = allPts[a][1];
        const bx = allPts[b][0], by = allPts[b][1];
        const cx = allPts[c][0], cy = allPts[c][1];
        if (triangleStrictlyInsidePolygon(ax, ay, bx, by, cx, cy, perimeterPts)) {
            validTris.push([a, b, c]);
        }
    }
    const edgeSet = new Set();
    const addEdge = (u, v) => {
        const key = u < v ? `${u},${v}` : `${v},${u}`;
        edgeSet.add(key);
    };
    for (const [a, b, c] of validTris) {
        addEdge(a, b); addEdge(b, c); addEdge(a, c);
    }
    for (let i = 0; i < nPerim; i++) {
        const j = (i + 1) % nPerim;
        addEdge(i, j);
    }
    const edges = [...edgeSet].map(k => k.split(',').map(Number));
    const zValues = allPts.map(() => 0.0);
    for (let i = 0; i < nPerim; i++) {
        zValues[i] = perimeterPts[i][2] || 0.0;
    }
    for (let i = 0; i < nInternal; i++) {
        zValues[nPerim + i] = internalSupportPts[i][2] || 0.0;
    }
    const seedPts = [
        ...perimeterPts.map(p => ({ x: p[0], y: p[1], z: p[2] || 0 })),
        ...internalSupportPts.map(p => ({ x: p[0], y: p[1], z: p[2] || 0 }))
    ];
    for (let i = nPerim + nInternal; i < allPts.length; i++) {
        const px = allPts[i][0], py = allPts[i][1];
        let wSum = 0, zSum = 0;
        for (const s of seedPts) {
            const d = Math.hypot(px - s.x, py - s.y);
            const w = 1.0 / Math.max(d, 1e-6) ** 2;
            wSum += w; zSum += w * s.z;
        }
        zValues[i] = wSum > 0 ? zSum / wSum : 0;
    }
    const nodes = allPts.map((p, i) => [p[0], p[1], zValues[i]]);
    const fixedIndices = new Set();
    for (let i = 0; i < nPerim + nInternal; i++) fixedIndices.add(i);
    const dx = Lx / Math.max(nx, 1);
    const dy = Ly / Math.max(ny, 1);
    return { nodes, edges, triangles: validTris, fixedIndices, xmin, xmax, ymin, ymax, Lx, Ly, dx, dy };
}

/**
 * Sample intermediate points along external line supports so they become
 * fixed nodes on the polygon mesh (used when add_line_support_3d alone is
 * not enough because the mesh was built without those coordinates).
 */
function sampleLineSupportPoints(lineSupports, spacingMm) {
    spacingMm = spacingMm || 500;
    const pts = [];
    for (const ls of lineSupports) {
        const x1 = parseFloat(ls.x1) || 0, y1 = parseFloat(ls.y1) || 0, z1 = parseFloat(ls.z1) || 0;
        const x2 = parseFloat(ls.x2) || 0, y2 = parseFloat(ls.y2) || 0, z2 = parseFloat(ls.z2) || 0;
        const len = Math.hypot(x2 - x1, y2 - y1, z2 - z1);
        const n = Math.max(2, Math.round(len / spacingMm));
        for (let i = 0; i <= n; i++) {
            const t = i / n;
            pts.push({
                x: x1 + t * (x2 - x1),
                y: y1 + t * (y2 - y1),
                z: z1 + t * (z2 - z1),
                dofs: (ls.dofs || "xyz").toLowerCase()
            });
        }
    }
    return pts;
}

// ---------------------------------------------------------------------------
// WORKER INIT
// ---------------------------------------------------------------------------
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
// ---------------------------------------------------------------------------
// MAIN MESSAGE HANDLER
// ---------------------------------------------------------------------------
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
        const domainMode = payload.domain_mode || "rectangular";
        // -----------------------------------------------------------------------
        // POLYGON MODE
        // -----------------------------------------------------------------------
        if (domainMode === "polygon") {
            try {
                const matType = payload.material_type || "membrane";
                const Lz_val  = parseFloat(payload.Lz) || 1000;
                const nx_req  = parseInt(payload.nx)   || 24;
                const ny_req  = parseInt(payload.ny)   || 24;
                const perimeterPts  = payload.perimeter_supports  || [];
                let internalPts     = payload.internal_supports   || [];
                const externalLines = payload.line_supports       || [];
                const internalLines = payload.internal_line_supports || [];

                if (perimeterPts.length < 3) {
                    postMessage({ status: "error", message: "At least 3 perimeter supports are required." });
                    return;
                }
                const area = polygonSignedArea(perimeterPts.map(p => [p.x, p.y]));
                if (Math.abs(area) < 1.0) {
                    postMessage({ status: "error", message: "Perimeter polygon is degenerate (zero area). Please add more distinct supports." });
                    return;
                }
                const pPts2D = perimeterPts.map(p => [p.x, p.y]);
                if (!polygonIsSimple(pPts2D)) {
                    postMessage({ status: "error", message: "Perimeter polygon is self-intersecting. Please enter perimeter supports in clockwise or anti-clockwise order (in plan, X-Y) so the outline has no crossing edges." });
                    return;
                }

                // Merge any residual internal-line samples (frontend usually already expands these)
                if (internalLines.length > 0) {
                    const sampled = sampleLineSupportPoints(internalLines, 500);
                    internalPts = internalPts.concat(sampled);
                }

                // External line supports: sample endpoints + intermediates into the
                // internal fixed set so the Delaunay mesh includes those nodes.
                // They will also be re-applied via add_line_support_3d in Python.
                if (externalLines.length > 0) {
                    const sampledExt = sampleLineSupportPoints(externalLines, 500);
                    internalPts = internalPts.concat(sampledExt);
                }

                const mesh = buildPolygonMesh(
                    perimeterPts.map(p  => [p.x, p.y, p.z || 0]),
                    internalPts.map(p   => [p.x, p.y, p.z || 0]),
                    nx_req, ny_req, Lz_val
                );
                if (!mesh.nodes.length || !mesh.edges.length) {
                    postMessage({ status: "error", message: "Polygon mesh generation failed — no valid triangles found inside the perimeter. Check that the perimeter is a valid simple polygon." });
                    return;
                }
                const meshJson    = JSON.stringify({
                    nodes:   mesh.nodes,
                    edges:   mesh.edges,
                    fixed:   [...mesh.fixedIndices],
                    xmin:    mesh.xmin, xmax: mesh.xmax,
                    ymin:    mesh.ymin, ymax: mesh.ymax,
                    Lx:      mesh.Lx,  Ly: mesh.Ly,
                    Lz:      Lz_val,
                    nx:      nx_req,   ny: ny_req,
                    dx:      mesh.dx,  dy: mesh.dy,
                    material_type: matType
                });
                const payloadJson = JSON.stringify(payload);
                const trianglesJson = JSON.stringify(mesh.triangles);
                const lineSupportsJson = JSON.stringify(externalLines);
                pyodide.globals.set("mesh_json_str",     meshJson);
                pyodide.globals.set("payload_json_str",  payloadJson);
                pyodide.globals.set("triangles_json_str", trianglesJson);
                pyodide.globals.set("line_supports_json_str", lineSupportsJson);
                const resultJson = await pyodide.runPythonAsync(`
import json, numpy as np
from core.domain_ff import FormFindingDomain3D
from core.materials import FormFindingMaterialRegistry
from core.solvers_ff import FormFindingSolverFactory
mesh_data    = json.loads(mesh_json_str)
payload      = json.loads(payload_json_str)
triangles_in = json.loads(triangles_json_str)
line_supports_in = json.loads(line_supports_json_str)
# Resolve material
mat_type = str(payload.get("material_type") or "membrane").lower()
if "cable" in mat_type or "rope" in mat_type:
    mat_type = "cables"
elif "fabric" in mat_type or "ptfe" in mat_type or "membrane" in mat_type:
    mat_type = "membrane"
payload["material_type"] = mat_type
mat_props = FormFindingMaterialRegistry.resolve_properties(payload)
# Build pre-wired domain (build_topology=False)
domain = FormFindingDomain3D(
    xmin=mesh_data["xmin"], xmax=mesh_data["xmax"],
    ymin=mesh_data["ymin"], ymax=mesh_data["ymax"],
    Lz=mesh_data["Lz"],
    nx=mesh_data["nx"], ny=mesh_data["ny"],
    geometry_preset="surface_grid",
    material_type=mat_type,
    build_topology=False
)
domain.nodes      = np.array(mesh_data["nodes"], dtype=float)
domain.edges      = np.array(mesh_data["edges"], dtype=int)
domain.fixed_nodes = set(mesh_data["fixed"])
domain.triangles  = triangles_in
domain.Lx = mesh_data["Lx"]; domain.Ly = mesh_data["Ly"]
domain.xmin = mesh_data["xmin"]; domain.xmax = mesh_data["xmax"]
domain.ymin = mesh_data["ymin"]; domain.ymax = mesh_data["ymax"]
domain.dx = mesh_data["dx"]; domain.dy = mesh_data["dy"]
# Apply external line supports as continuous 3D restraints (snaps nearby mesh nodes)
for ls in line_supports_in:
    p1 = (float(ls.get("x1", 0)), float(ls.get("y1", 0)), float(ls.get("z1", 0)))
    p2 = (float(ls.get("x2", 0)), float(ls.get("y2", 0)), float(ls.get("z2", 0)))
    if hasattr(domain, "add_line_support_3d"):
        domain.add_line_support_3d(p1, p2)
# Prestress
prestress_warp_N_mm = float(payload.get("prestress_warp_kn_m", 2.0))
prestress_weft_N_mm = float(payload.get("prestress_weft_kn_m", 2.0))
edge_cable_prestress_N = float(payload.get("edge_cable_prestress_kn", 20.0)) * 1000.0
t_mm = max(float(payload.get("sec_fabric_t", 1.2)), 0.1)
area_mm2 = t_mm * 1000.0
gamma = mat_props.get("gamma_kn_m3", 25.0) if bool(payload.get("include_self_weight", True)) else 0.0
solver = FormFindingSolverFactory.create(
    material_type          = mat_type,
    domain                 = domain,
    mat_props              = mat_props,
    gamma_kn_m3            = gamma,
    area_mm2               = area_mm2,
    prestress_force        = 0.0,
    prestress_warp_N_mm    = prestress_warp_N_mm,
    prestress_weft_N_mm    = prestress_weft_N_mm,
    edge_cable_prestress_N = edge_cable_prestress_N,
    point_loads            = payload.get("loads", [])
)
equilibrium_nodes, axial_forces, reactions, diagnostics = solver.solve_equilibrium(
    iterations=int(payload.get("max_iterations", 20000)), rel_tol=1e-4
)
displacement_vecs  = equilibrium_nodes - np.copy(domain.nodes).astype(float)
deflections_mm     = np.linalg.norm(displacement_vecs, axis=1)
u_max              = float(np.max(deflections_mm)) if len(deflections_mm) > 0 else 0.0
element_stresses   = axial_forces / max(area_mm2, 1e-4)
num_nodes = len(equilibrium_nodes)
nodal_stresses = np.zeros(num_nodes, dtype=float)
node_degree    = np.zeros(num_nodes, dtype=float)
for i, (u, v) in enumerate(domain.edges):
    s = element_stresses[i]
    nodal_stresses[u] += s; nodal_stresses[v] += s
    node_degree[u] += 1.0; node_degree[v] += 1.0
node_degree = np.maximum(node_degree, 1.0)
nodal_stresses /= node_degree
fixed_indices = sorted(list(domain.fixed_nodes))
reaction_data = []
for idx in fixed_indices:
    pos = [float(v) for v in equilibrium_nodes[idx].tolist()]
    rx, ry, rz = [float(v) for v in reactions[idx].tolist()]
    reaction_data.append({
        "node": int(idx), "pos": pos,
        "Rx_kN": round(rx/1000, 3), "Ry_kN": round(ry/1000, 3), "Rz_kN": round(rz/1000, 3),
        "R_total_kN": round(float(np.linalg.norm([rx,ry,rz]))/1000, 3)
    })
json.dumps({
    "nodes":          [[float(v) for v in row] for row in equilibrium_nodes.tolist()],
    "edges":          [[int(v)   for v in row] for row in domain.edges.tolist()],
    "triangles":      triangles_in,
    "axial_forces":   [float(v) for v in axial_forces.tolist()],
    "stresses_mpa":   [float(v) for v in nodal_stresses.tolist()],
    "deflections_mm": [float(v) for v in deflections_mm.tolist()],
    "sigma_max_tens": round(float(np.max(nodal_stresses)), 3),
    "sigma_max_comp": round(float(np.min(nodal_stresses)), 3),
    "u_max":          round(u_max, 3),
    "reactions":      reaction_data,
    "material":       mat_props.get("material_name", mat_type),
    "num_nodes":      num_nodes,
    "num_edges":      len(domain.edges),
    "nx_actual":      mesh_data["nx"],
    "ny_actual":      mesh_data["ny"],
    "domain_mode":    "polygon",
    "diagnostics":    diagnostics,
})
`);
                postMessage({ status: "completed", data: JSON.parse(resultJson) });
            } catch (err) {
                console.error("[worker_ff.js] Polygon solve failed:", err);
                postMessage({ status: "error", message: err.toString() });
            }
            return;
        }
        // -----------------------------------------------------------------------
        // RECTANGULAR MODE (legacy — unchanged)
        // -----------------------------------------------------------------------
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
# --- CRITICAL FIX: compute the TRUE bounding box from every declared support ---
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
forced_x = [float(pt.get("x", 0)) for pt in point_sups]
forced_y = [float(pt.get("y", 0)) for pt in point_sups]
for l_sup in line_sups:
    forced_x.append(float(l_sup.get("x1", 0)))
    forced_x.append(float(l_sup.get("x2", 0)))
    forced_y.append(float(l_sup.get("y1", 0)))
    forced_y.append(float(l_sup.get("y2", 0)))
_ny = int(payload.get("ny", 12))
if mat_type in ("cables", "cable"):
    _ny = 1
domain = FormFindingDomain3D(
    xmin=xmin_val,
    xmax=xmax_val,
    ymin=ymin_val,
    ymax=ymax_val,
    Lz=Lz_val,
    nx=int(payload.get("nx", 36)),
    ny=_ny,
    forced_x=forced_x,
    forced_y=forced_y,
    geometry_preset="surface_grid",
    material_type=mat_type
)
# ------------------------------------------------------------------
# CABLE SPECIAL CASE – force a pure 1-D polyline between the two
# supports.  This completely overrides the surface_grid so that
# ForceDensity always solves a single chain that reaches the supports.
# ------------------------------------------------------------------
if mat_type in ("cables", "cable") and len(point_sups) >= 2:
    pA = point_sups[0]
    pB = point_sups[1]
    # order by dominant plan direction so ratio 0 → start is consistent
    if abs(float(pB.get("x",0)) - float(pA.get("x",0))) >= abs(float(pB.get("y",0)) - float(pA.get("y",0))):
        if float(pA.get("x",0)) > float(pB.get("x",0)):
            pA, pB = pB, pA
    else:
        if float(pA.get("y",0)) > float(pB.get("y",0)):
            pA, pB = pB, pA
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
    # overwrite domain geometry
    domain.nodes = nodes
    domain.edges = edges
    domain.fixed_nodes = {0, nx}          # the two ends
    domain.nx = nx
    domain.ny = 1
    # keep xmin/xmax etc. for any downstream code that reads them
    domain.xmin, domain.xmax = min(x1, x2), max(x1, x2)
    domain.ymin, domain.ymax = min(y1, y2), max(y1, y2)
# Discrete Point Supports (only needed for non-cable cases now)
seed_z_map = {}
if mat_type not in ("cables", "cable"):
    for pt in point_sups:
        px, py, pz = float(pt.get("x", 0)), float(pt.get("y", 0)), float(pt.get("z", 0))
        domain.add_point_support(px, py, pz)
        seed_z_map[(px, py)] = pz
    for l_sup in line_sups:
        p1 = (float(l_sup.get("x1", 0)), float(l_sup.get("y1", 0)), float(l_sup.get("z1", 0)))
        p2 = (float(l_sup.get("x2", 0)), float(l_sup.get("y2", 0)), float(l_sup.get("z2", 0)))
        if hasattr(domain, 'add_line_support_3d'):
            domain.add_line_support_3d(p1, p2)
        if (p1[0], p1[1]) not in seed_z_map:
            seed_z_map[(p1[0], p1[1])] = p1[2]
        if (p2[0], p2[1]) not in seed_z_map:
            seed_z_map[(p2[0], p2[1])] = p2[2]
    # Fallback presets
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
# Seed interior elevations (skip for cables – already done by the polyline)
if hasattr(domain, 'apply_idw_surface_interpolation') and mat_type not in ("cables", "cable"):
    domain.apply_idw_surface_interpolation(seed_z_map)
# Material Cross-Section Area & Prestress Parsing
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
    area_mm2 = t_mm * 1000.0
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
    iterations = int(payload.get("max_iterations", 20000)),
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
# --- CRITICAL FIX: report the ACTUAL solved grid resolution -----------------
nx_actual_val = int(getattr(domain, "nx", payload.get("nx", 36)))
ny_actual_val = int(getattr(domain, "ny", payload.get("ny", 12)))
json.dumps({
    "nodes":          [[float(v) for v in row] for row in equilibrium_nodes.tolist()],
    "edges":          [[int(v)   for v in row] for row in np.asarray(domain.edges, dtype=int).tolist()],
    "triangles":      None,
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
    "nx_actual":      nx_actual_val,
    "ny_actual":      ny_actual_val,
    "domain_mode":    "rectangular",
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
