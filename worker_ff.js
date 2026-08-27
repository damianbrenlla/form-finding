// worker_ff.js
importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;

async function initPyodideRuntime() {
    try {
        self.postMessage({ status: 'log', message: 'Downloading Pyodide core & WebAssembly engine...' });
        pyodide = await loadPyodide();
        
        self.postMessage({ status: 'log', message: 'Compiling NumPy vectorisation library...' });
        await pyodide.loadPackage(["numpy"]);

        self.postMessage({ status: 'log', message: 'Injecting DBSW structural FE classes...' });
        
        // 1. Inject FormFindingDomain3D and Solver classes into Pyodide global scope
        await pyodide.runPythonAsync(`
import json
import numpy as np

# ------------------------------------------------------------------
# DBSW Spatial Form-Finding Network Domain
# ------------------------------------------------------------------
class FormFindingDomain3D:
    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        Lz: float,
        nx: int,
        ny: int,
        forced_x=None,
        forced_y=None,
        geometry_preset: str = "surface_grid",
        material_type: str = "cables"
    ):
        self.xmin = float(xmin)
        self.xmax = float(xmax) if xmax > xmin else float(xmin) + 1.0
        self.ymin = float(ymin)
        self.ymax = float(ymax) if ymax > ymin else float(ymin) + 1.0
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.Lz = float(Lz)
        self.nx = int(max(nx, 2))
        self.ny = int(max(ny, 2))
        self.geometry_preset = geometry_preset
        self.material_type = str(material_type).lower()

        self.nodes = []
        self.edges = []
        self.triangles = np.empty((0, 3), dtype=int)
        self.boundary_edges = np.empty((0, 2), dtype=int)
        self.boundary_node_indices = set()
        self.fixed_nodes = set()
        self.node_loads = {}

        self._x_lin = None
        self._y_lin = None
        self.perimeter_xy = None
        self.interior_xy = None

        self._build_network_topology(forced_x or [], forced_y or [])

        self.dx = float(np.mean(np.diff(self._x_lin))) if self._x_lin is not None and len(self._x_lin) > 1 else 0.0
        self.dy = float(np.mean(np.diff(self._y_lin))) if self._y_lin is not None and len(self._y_lin) > 1 else 0.0

    @staticmethod
    def _point_on_segment(p, a, b, tol=1e-8):
        p = np.asarray(p, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < tol * tol:
            return np.linalg.norm(p - a) <= tol
        t = float(np.dot(p - a, ab) / denom)
        if t < -tol or t > 1.0 + tol:
            return False
        proj = a + np.clip(t, 0.0, 1.0) * ab
        return np.linalg.norm(p - proj) <= tol

    @classmethod
    def _point_in_polygon(cls, point, polygon, tol=1e-7):
        p = np.asarray(point, dtype=float)
        inside = False
        n = len(polygon)
        for i in range(n):
            a = polygon[i]
            b = polygon[(i + 1) % n]
            if cls._point_on_segment(p, a, b, tol):
                return True
            xi, yi = a
            xj, yj = b
            crosses = ((yi > p[1]) != (yj > p[1]))
            if crosses:
                x_at_y = (xj - xi) * (p[1] - yi) / (yj - yi + 1e-30) + xi
                if p[0] < x_at_y:
                    inside = not inside
        return inside

    @staticmethod
    def _orientation(a, b, c):
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @classmethod
    def _proper_segment_intersection(cls, a, b, c, d, tol=1e-8):
        o1 = cls._orientation(a, b, c)
        o2 = cls._orientation(a, b, d)
        o3 = cls._orientation(c, d, a)
        o4 = cls._orientation(c, d, b)

        if abs(o1) <= tol and cls._point_on_segment(c, a, b, tol):
            return False
        if abs(o2) <= tol and cls._point_on_segment(d, a, b, tol):
            return False
        if abs(o3) <= tol and cls._point_on_segment(a, c, d, tol):
            return False
        if abs(o4) <= tol and cls._point_on_segment(b, c, d, tol):
            return False

        return ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and \\
               ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol))

    @classmethod
    def _triangle_is_inside_polygon(cls, tri_xy, polygon):
        centroid = np.mean(tri_xy, axis=0)
        if not cls._point_in_polygon(centroid, polygon):
            return False
        for p in tri_xy:
            if not cls._point_in_polygon(p, polygon):
                return False
        for i in range(3):
            midpoint = 0.5 * (tri_xy[i] + tri_xy[(i + 1) % 3])
            if not cls._point_in_polygon(midpoint, polygon):
                return False

        for i in range(3):
            a = tri_xy[i]
            b = tri_xy[(i + 1) % 3]
            for j in range(len(polygon)):
                c = polygon[j]
                d = polygon[(j + 1) % len(polygon)]
                if cls._proper_segment_intersection(a, b, c, d):
                    return False
        return True

    @classmethod
    def _order_boundary_segments(cls, line_segments, tol=1e-6):
        if len(line_segments) < 3:
            raise ValueError("Polygon membrane requires at least 3 External line supports.")

        segs = []
        for i, seg in enumerate(line_segments):
            p1 = np.asarray(seg[0], dtype=float)
            p2 = np.asarray(seg[1], dtype=float)
            if np.linalg.norm(p2[:2] - p1[:2]) < tol:
                raise ValueError(f"External line support {i + 1} has zero plan length.")
            segs.append((p1, p2))

        def same_xy(a, b):
            return np.linalg.norm(a[:2] - b[:2]) <= tol

        vertices = []
        incidences = []
        for si, (a, b) in enumerate(segs):
            ids = []
            for p in (a, b):
                found = None
                for vi, v in enumerate(vertices):
                    if same_xy(v, p):
                        found = vi
                        break
                if found is None:
                    vertices.append(p.copy())
                    incidences.append([])
                    found = len(vertices) - 1
                ids.append(found)
            incidences[ids[0]].append(si)
            incidences[ids[1]].append(si)

        for vi, inc in enumerate(incidences):
            if len(inc) != 2:
                raise ValueError(
                    "External line supports must form one closed polygon: "
                    f"boundary vertex ({vertices[vi][0]:.1f}, {vertices[vi][1]:.1f}) "
                    f"is connected to {len(inc)} line supports instead of 2."
                )

        ordered = [vertices[0].copy()]
        current_vertex = 0
        previous_segment = None
        used = set()

        while True:
            candidates = [si for si in incidences[current_vertex] if si != previous_segment and si not in used]
            if not candidates:
                break
            si = candidates[0]
            used.add(si)
            a, b = segs[si]
            next_point = b if same_xy(vertices[current_vertex], a) else a

            next_vertex = None
            for vi, v in enumerate(vertices):
                if same_xy(v, next_point):
                    next_vertex = vi
                    break
            if next_vertex is None:
                raise ValueError("Failed to connect External line support endpoints into a polygon.")

            if next_vertex == 0:
                break
            ordered.append(next_point.copy())
            previous_segment = si
            current_vertex = next_vertex

        if len(used) != len(segs) or len(ordered) < 3:
            raise ValueError(
                "External line supports do not form one closed polygon. "
                "Ensure every external line endpoint connects to exactly one other external line endpoint."
            )

        return np.asarray(ordered, dtype=float)

    @classmethod
    def build_polygon_domain_from_line_segments(
        cls,
        boundary_segments,
        interior_points=None,
        interior_line_segments=None,
        nx=36,
        ny=12,
        Lz=1000.0,
        material_type="membrane",
    ):
        perimeter = cls._order_boundary_segments(boundary_segments)
        return cls.build_polygon_domain(
            perimeter_points=perimeter.tolist(),
            interior_points=interior_points or [],
            interior_line_segments=interior_line_segments or [],
            nx=nx, ny=ny, Lz=Lz, material_type=material_type,
        )

    @classmethod
    def _delaunay_triangles(cls, points):
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n < 3:
            return np.empty((0, 3), dtype=int)

        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        span = max(max_x - min_x, max_y - min_y, 1.0)
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)

        super_pts = np.array([
            [cx - 20.0 * span, cy - 20.0 * span],
            [cx,               cy + 20.0 * span],
            [cx + 20.0 * span, cy - 20.0 * span],
        ], dtype=float)
        all_pts = np.vstack([pts, super_pts])
        st0, st1, st2 = n, n + 1, n + 2
        triangles = [(st0, st1, st2)]

        def circumcircle(tri):
            a, b, c = all_pts[list(tri)]
            ax, ay = a; bx, by = b; cx_, cy_ = c
            d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
            if abs(d) < 1e-18:
                return None
            a2 = ax * ax + ay * ay
            b2 = bx * bx + by * by
            c2 = cx_ * cx_ + cy_ * cy_
            ux = (a2 * (by - cy_) + b2 * (cy_ - ay) + c2 * (ay - by)) / d
            uy = (a2 * (cx_ - bx) + b2 * (ax - cx_) + c2 * (bx - ax)) / d
            centre = np.array([ux, uy])
            r2 = float(np.dot(centre - a, centre - a))
            return centre, r2

        circles = {triangles[0]: circumcircle(triangles[0])}
        eps = max(span * span * 1e-12, 1e-12)

        for pi in range(n):
            pnt = all_pts[pi]
            bad = []
            for tri in triangles:
                circle = circles.get(tri)
                if circle is None:
                    continue
                centre, r2 = circle
                d2 = float(np.dot(pnt - centre, pnt - centre))
                if d2 <= r2 + eps:
                    bad.append(tri)

            edge_count = {}
            for tri in bad:
                for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    key = tuple(sorted(edge))
                    edge_count[key] = edge_count.get(key, 0) + 1

            for tri in bad:
                if tri in circles:
                    del circles[tri]
                triangles.remove(tri)

            boundary = [edge for edge, count in edge_count.items() if count == 1]
            for a, b in boundary:
                tri = (a, b, pi)
                if cls._orientation(all_pts[a], all_pts[b], all_pts[pi]) < 0.0:
                    tri = (b, a, pi)
                if abs(cls._orientation(all_pts[tri[0]], all_pts[tri[1]], all_pts[tri[2]])) < 1e-18:
                    continue
                triangles.append(tri)
                circles[tri] = circumcircle(tri)

        result = [tri for tri in triangles if all(v < n for v in tri)]
        return np.asarray(result, dtype=int) if result else np.empty((0, 3), dtype=int)

    @classmethod
    def build_polygon_domain(
        cls,
        perimeter_points,
        interior_points=None,
        interior_line_segments=None,
        nx=36,
        ny=12,
        Lz=1000.0,
        material_type="membrane",
    ):
        interior_points = interior_points or []
        interior_line_segments = interior_line_segments or []

        if len(perimeter_points) < 3:
            raise ValueError("Polygon membrane requires at least 3 Perimeter support points.")

        perim3 = np.asarray(perimeter_points, dtype=float)
        if perim3.ndim != 2 or perim3.shape[1] != 3:
            raise ValueError("Perimeter support points must be [x, y, z] coordinates.")

        if len(perim3) >= 4 and np.linalg.norm(perim3[0, :2] - perim3[-1, :2]) < 1e-9:
            perim3 = perim3[:-1]

        if len(perim3) < 3:
            raise ValueError("Polygon membrane requires at least 3 distinct Perimeter points.")

        for i in range(len(perim3)):
            for j in range(i):
                if np.linalg.norm(perim3[i, :2] - perim3[j, :2]) < 1e-8:
                    raise ValueError(f"Duplicate Perimeter point at rows {j + 1} and {i + 1}.")

        polygon = perim3[:, :2].copy()

        for i in range(len(polygon)):
            a, b = polygon[i], polygon[(i + 1) % len(polygon)]
            for j in range(i + 1, len(polygon)):
                if j in (i, (i + 1) % len(polygon), (i - 1) % len(polygon)):
                    continue
                c, d = polygon[j], polygon[(j + 1) % len(polygon)]
                if cls._proper_segment_intersection(a, b, c, d):
                    raise ValueError("Perimeter polygon is self-intersecting. Check the Perimeter row order.")

        if abs(cls._polygon_signed_area(polygon)) < 1e-8:
            raise ValueError("Perimeter polygon has zero area.")

        interior3 = np.asarray(interior_points, dtype=float) if interior_points else np.empty((0, 3), dtype=float)
        if interior3.size and (interior3.ndim != 2 or interior3.shape[1] != 3):
            raise ValueError("Interior support points must be [x, y, z] coordinates.")

        for i, p in enumerate(interior3):
            if not cls._point_in_polygon(p[:2], polygon):
                raise ValueError(
                    f"Interior support {i + 1} at ({p[0]:.1f}, {p[1]:.1f}) lies outside the Perimeter polygon."
                )
            for k in range(len(polygon)):
                if cls._point_on_segment(p[:2], polygon[k], polygon[(k + 1) % len(polygon)], 1e-7):
                    raise ValueError(
                        f"Interior support {i + 1} lies on the Perimeter. "
                        "Change its Role to Perimeter."
                    )

        internal_line_pts = []
        for li, seg in enumerate(interior_line_segments):
            if len(seg) != 2:
                raise ValueError(f"Interior line support {li + 1} is invalid.")
            p1 = np.asarray(seg[0], dtype=float)
            p2 = np.asarray(seg[1], dtype=float)
            if np.linalg.norm(p2[:2] - p1[:2]) < 1e-8:
                raise ValueError(f"Interior line support {li + 1} has zero plan length.")
            for p in (p1, p2):
                if not cls._point_in_polygon(p[:2], polygon):
                    raise ValueError(
                        f"Interior line support {li + 1} endpoint at ({p[0]:.1f}, {p[1]:.1f}) lies outside the Perimeter polygon."
                    )
            line_len = np.linalg.norm(p2[:2] - p1[:2])
            target_spacing = min(
                ((np.max(polygon[:, 0]) - np.min(polygon[:, 0])) / max(int(nx), 2)),
                ((np.max(polygon[:, 1]) - np.min(polygon[:, 1])) / max(int(ny), 2))
            )
            nseg = max(1, int(np.ceil(line_len / max(target_spacing, 1e-6))))
            nseg = min(nseg, 150)
            for k in range(nseg + 1):
                t = k / nseg
                internal_line_pts.append(p1 + t * (p2 - p1))

        obj = cls.__new__(cls)
        obj.xmin = float(np.min(polygon[:, 0]))
        obj.xmax = float(np.max(polygon[:, 0]))
        obj.ymin = float(np.min(polygon[:, 1]))
        obj.ymax = float(np.max(polygon[:, 1]))
        obj.Lx = obj.xmax - obj.xmin
        obj.Ly = obj.ymax - obj.ymin
        obj.Lz = float(Lz)
        obj.nx = int(max(nx, 2))
        obj.ny = int(max(ny, 2))
        obj.geometry_preset = "polygon"
        obj.material_type = str(material_type).lower()
        obj.nodes = []
        obj.edges = []
        obj.triangles = np.empty((0, 3), dtype=int)
        obj.boundary_edges = np.empty((0, 2), dtype=int)
        obj.boundary_node_indices = set()
        obj.fixed_nodes = set()
        obj.node_loads = {}
        obj.perimeter_xy = polygon.copy()
        obj.interior_xy = interior3[:, :2].copy() if len(interior3) else np.empty((0, 2))
        obj._x_lin = np.linspace(obj.xmin, obj.xmax, obj.nx + 1)
        obj._y_lin = np.linspace(obj.ymin, obj.ymax, obj.ny + 1)

        candidates = []
        for x in obj._x_lin:
            for y in obj._y_lin:
                if cls._point_in_polygon((x, y), polygon):
                    candidates.append([x, y, 0.0])

        def add_unique_xy(p3):
            p2 = np.asarray(p3[:2], dtype=float)
            for existing in candidates:
                if np.linalg.norm(np.asarray(existing[:2]) - p2) < 1e-8:
                    return
            candidates.append([float(p3[0]), float(p3[1]), float(p3[2])])

        for p in perim3:
            add_unique_xy(p)
        for p in interior3:
            add_unique_xy(p)
        for p in internal_line_pts:
            add_unique_xy(p)

        xy = np.asarray(candidates, dtype=float)[:, :2]
        if len(xy) < 3:
            raise ValueError("Polygon mesh generation produced fewer than 3 mesh vertices.")

        try:
            delaunay_triangles = cls._delaunay_triangles(xy)
        except Exception as exc:
            raise ValueError(f"Polygon triangulation failed: {exc}")

        valid_triangles = []
        for simplex in delaunay_triangles:
            tri_xy = xy[simplex]
            if cls._triangle_is_inside_polygon(tri_xy, polygon):
                valid_triangles.append([int(simplex[0]), int(simplex[1]), int(simplex[2])])

        if not valid_triangles:
            raise ValueError("Polygon triangulation produced no valid interior triangles.")

        nodes = np.zeros((len(candidates), 3), dtype=float)
        nodes[:, :2] = xy
        for p in perim3:
            idx = int(np.argmin(np.linalg.norm(xy - p[:2], axis=1)))
            nodes[idx] = p
        for p in interior3:
            idx = int(np.argmin(np.linalg.norm(xy - p[:2], axis=1)))
            nodes[idx] = p

        triangles = np.asarray(valid_triangles, dtype=int)

        edge_set = set()
        for tri in triangles:
            a, b, c = map(int, tri)
            edge_set.add(tuple(sorted((a, b))))
            edge_set.add(tuple(sorted((b, c))))
            edge_set.add(tuple(sorted((c, a))))

        boundary_edges = []
        for i in range(len(perim3)):
            p = perim3[i, :2]
            q = perim3[(i + 1) % len(perim3), :2]
            ip = int(np.argmin(np.linalg.norm(xy - p, axis=1)))
            iq = int(np.argmin(np.linalg.norm(xy - q, axis=1)))
            edge = tuple(sorted((ip, iq)))
            edge_set.add(edge)
            boundary_edges.append(edge)

        for seg in interior_line_segments:
            p1 = np.asarray(seg[0], dtype=float)
            p2 = np.asarray(seg[1], dtype=float)
            line_len = np.linalg.norm(p2[:2] - p1[:2])
            target_spacing = min(
                obj.Lx / max(obj.nx, 2),
                obj.Ly / max(obj.ny, 2),
                max(line_len, 1e-6)
            )
            nseg = max(1, min(150, int(np.ceil(line_len / max(target_spacing, 1e-6)))))
            line_indices = []
            for k in range(nseg + 1):
                t = k / nseg
                pt = p1 + t * (p2 - p1)
                idx = int(np.argmin(np.linalg.norm(xy - pt[:2], axis=1)))
                line_indices.append(idx)
            for a, b in zip(line_indices[:-1], line_indices[1:]):
                if a != b:
                    edge_set.add(tuple(sorted((a, b))))

        obj.nodes = nodes
        obj.triangles = triangles
        obj.edges = np.asarray(sorted(edge_set), dtype=int)
        obj.boundary_edges = np.asarray(boundary_edges, dtype=int)
        obj.boundary_node_indices = set(int(v) for edge in boundary_edges for v in edge)

        for p in perim3:
            idx = int(np.argmin(np.linalg.norm(obj.nodes[:, :2] - p[:2], axis=1)))
            obj.fixed_nodes.add(idx)
        for p in interior3:
            idx = int(np.argmin(np.linalg.norm(obj.nodes[:, :2] - p[:2], axis=1)))
            obj.fixed_nodes.add(idx)
        for p in internal_line_pts:
            idx = int(np.argmin(np.linalg.norm(obj.nodes[:, :2] - p[:2], axis=1)))
            obj.fixed_nodes.add(idx)

        obj.dx = float(np.mean(np.diff(obj._x_lin))) if len(obj._x_lin) > 1 else 0.0
        obj.dy = float(np.mean(np.diff(obj._y_lin))) if len(obj._y_lin) > 1 else 0.0
        return obj

    @staticmethod
    def _polygon_signed_area(polygon):
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    def _build_network_topology(self, forced_x, forced_y):
        is_pure_cable = self.material_type in ("cables", "cable")

        if is_pure_cable:
            x_lin = np.linspace(self.xmin, self.xmax, self.nx + 1)
            self._x_lin = x_lin
            self._y_lin = np.array([self.ymin, self.ymax], dtype=float)
            for i in range(self.nx + 1):
                self.nodes.append([x_lin[i], self.ymin, 0.0])
                if i < self.nx:
                    self.edges.append([i, i + 1])
        else:
            x_lin = np.linspace(self.xmin, self.xmax, self.nx + 1)
            y_lin = np.linspace(self.ymin, self.ymax, self.ny + 1)
            if len(forced_x) > 0:
                x_lin = np.union1d(x_lin, np.asarray(forced_x, dtype=float))
            if len(forced_y) > 0:
                y_lin = np.union1d(y_lin, np.asarray(forced_y, dtype=float))
            self._x_lin, self._y_lin = x_lin, y_lin

            grid_map = {}
            node_idx = 0
            for i, x in enumerate(x_lin):
                for j, y in enumerate(y_lin):
                    self.nodes.append([x, y, 0.0])
                    grid_map[(i, j)] = node_idx
                    node_idx += 1

            nx_actual = len(x_lin) - 1
            ny_actual = len(y_lin) - 1
            triangles = []
            for i in range(nx_actual + 1):
                for j in range(ny_actual + 1):
                    curr = grid_map[(i, j)]
                    if i < nx_actual:
                        self.edges.append([curr, grid_map[(i + 1, j)]])
                    if j < ny_actual:
                        self.edges.append([curr, grid_map[(i, j + 1)]])
                    if i < nx_actual and j < ny_actual:
                        n_x = grid_map[(i + 1, j)]
                        n_y = grid_map[(i, j + 1)]
                        n_xy = grid_map[(i + 1, j + 1)]
                        self.edges.append([curr, n_xy])
                        self.edges.append([n_x, n_y])
                        triangles.append([curr, n_x, n_xy])
                        triangles.append([curr, n_xy, n_y])

            self.triangles = np.asarray(triangles, dtype=int)
            self.nx = nx_actual
            self.ny = ny_actual

        self.nodes = np.array(self.nodes, dtype=float)
        self.edges = np.array(self.edges, dtype=int)
        if self.nodes.ndim == 1:
            self.nodes = self.nodes.reshape((-1, 3))
        if self.edges.size == 0:
            self.edges = np.empty((0, 2), dtype=int)

    def _auto_tolerance(self) -> float:
        dx = self.Lx / self.nx if self.nx > 0 else 100.0
        dy = self.Ly / self.ny if self.ny > 0 else 100.0
        return max(dx, dy) * 0.51

    def apply_idw_surface_interpolation(self, seed_z_map: dict, power: float = 2.0):
        if self.material_type in ("cables", "cable") or len(self.nodes) == 0:
            return
        if not seed_z_map:
            return

        pts = np.array(list(seed_z_map.keys()), dtype=float)
        zvals = np.array(list(seed_z_map.values()), dtype=float)

        free_mask = np.ones(len(self.nodes), dtype=bool)
        if self.fixed_nodes:
            free_mask[list(self.fixed_nodes)] = False

        free_xy = self.nodes[free_mask, :2]
        if len(free_xy) == 0:
            return

        diffs = free_xy[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        weights = 1.0 / np.maximum(dists, 1e-6) ** power
        z_interp = (weights @ zvals) / np.sum(weights, axis=1)

        exact_hit = dists < 1e-6
        any_exact = np.any(exact_hit, axis=1)
        if np.any(any_exact):
            for row in np.where(any_exact)[0]:
                z_interp[row] = zvals[np.argmax(exact_hit[row])]

        self.nodes[free_mask, 2] = z_interp

    def add_point_support(self, x: float, y: float, z: float, tol: float = None):
        if tol is None:
            tol = self._auto_tolerance()

        pt_2d = np.array([float(x), float(y)])
        dists_2d = np.linalg.norm(self.nodes[:, :2] - pt_2d, axis=1)
        idx = int(np.argmin(dists_2d))

        if dists_2d[idx] <= tol or self.material_type in ("cables", "cable"):
            self.fixed_nodes.add(idx)
            self.nodes[idx] = [float(x), float(y), float(z)]
            return idx

        raise ValueError(
            f"Point support at ({x}, {y}) is {dists_2d[idx]:.2f}mm from the nearest "
            f"mesh node — outside tolerance ({tol:.2f}mm). It was NOT attached."
        )

    def add_line_support(self, axis: str = "x", value: float = 0.0, tol: float = None):
        if tol is None:
            tol = self._auto_tolerance()

        axis_map = {"x": 0, "y": 1, "z": 2}
        col = axis_map[axis.lower()]
        matches = np.where(np.abs(self.nodes[:, col] - value) <= tol)[0]
        for idx in matches:
            self.fixed_nodes.add(int(idx))

    def add_line_support_3d(self, p1: tuple, p2: tuple, tol: float = None, boundary_only: bool = None):
        p1_arr = np.array(p1, dtype=float)
        p2_arr = np.array(p2, dtype=float)
        line_vec = p2_arr - p1_arr
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-3:
            self.add_point_support(p1_arr[0], p1_arr[1], p1_arr[2], tol)
            return

        line_dir = line_vec / line_len
        dx = self.Lx / self.nx if self.nx > 0 else 100.0
        dy = self.Ly / self.ny if self.ny > 0 else 100.0
        perp_x = -line_dir[1]
        perp_y = line_dir[0]
        directional_spacing = np.sqrt((dx * perp_x) ** 2 + (dy * perp_y) ** 2)
        effective_tol = max(directional_spacing * 1.25, 10.0) if tol is None else tol

        if boundary_only is None:
            boundary_only = self.geometry_preset != "polygon" and self.material_type not in ("cables", "cable")
        candidate_indices = self.get_boundary_nodes() if boundary_only else range(len(self.nodes))

        for idx in candidate_indices:
            node = self.nodes[idx]
            node_vec = node - p1_arr
            proj_len = np.dot(node_vec, line_dir)
            if -effective_tol <= proj_len <= line_len + effective_tol:
                proj_pt = p1_arr + np.clip(proj_len, 0.0, line_len) * line_dir
                dist = np.linalg.norm(node - proj_pt)
                if dist <= effective_tol:
                    already_fixed = idx in self.fixed_nodes
                    self.fixed_nodes.add(idx)
                    if not already_fixed:
                        self.nodes[idx] = proj_pt

    def add_edge_support(self, edge: str = "all"):
        tol = self._auto_tolerance()
        if self.geometry_preset == "polygon":
            if edge == "all":
                self.fixed_nodes.update(self.boundary_node_indices)
            return
        if edge in ("all", "x0"):
            self.add_line_support("x", self.xmin, tol)
        if edge in ("all", "xmax"):
            self.add_line_support("x", self.xmax, tol)
        if edge in ("all", "y0"):
            self.add_line_support("y", self.ymin, tol)
        if edge in ("all", "ymax"):
            self.add_line_support("y", self.ymax, tol)

    def get_boundary_nodes(self) -> np.ndarray:
        if self.geometry_preset == "polygon":
            return np.asarray(sorted(self.boundary_node_indices), dtype=int)

        dx = self.Lx / self.nx if self.nx > 0 else 100.0
        dy = self.Ly / self.ny if self.ny > 0 else 100.0
        tol = max(dx, dy) * 0.51
        mask = (
            (np.abs(self.nodes[:, 0] - self.xmin) < tol)
            | (np.abs(self.nodes[:, 0] - self.xmax) < tol)
            | (np.abs(self.nodes[:, 1] - self.ymin) < tol)
            | (np.abs(self.nodes[:, 1] - self.ymax) < tol)
        )
        return np.where(mask)[0]

    def get_stats(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_triangles": int(len(self.triangles)),
            "num_fixed": len(self.fixed_nodes),
            "grid_spacing_x": self.dx,
            "grid_spacing_y": self.dy,
            "xmin": self.xmin, "xmax": self.xmax,
            "ymin": self.ymin, "ymax": self.ymax,
            "geometry": self.geometry_preset,
        }
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

            let pyExecutionScript = `
import json
import numpy as np

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

            let jsonResult = await pyodide.runPythonAsync(pyExecutionScript);
            self.postMessage({ status: 'completed', data: JSON.parse(jsonResult) });

        } catch (err) {
            self.postMessage({ status: 'error', message: err.message });
        }
    }
};
