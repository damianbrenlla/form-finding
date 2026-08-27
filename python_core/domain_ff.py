# DBSW Spatial Form-Finding Network Domain
# Author: Damian Brenlla / DBSW 2026
# v14 — Polygon Domain Engine: Arbitrary planar perimeter polylines, explicit interior/line supports,
#       Point-in-Polygon ray-casting, and Delaunay boundary-edge recovery pass.

import numpy as np
from scipy.spatial import Delaunay


class FormFindingDomain3D:
    def __init__(
        self,
        nodes: np.ndarray,
        edges: np.ndarray,
        faces: np.ndarray,
        fixed_nodes: set,
        perimeter_nodes: set,
        material_type: str = "membrane",
        Lx: float = 6000.0,
        Ly: float = 3000.0,
        Lz: float = 1000.0
    ):
        self.nodes = np.array(nodes, dtype=float)
        self.edges = np.array(edges, dtype=int)
        self.faces = np.array(faces, dtype=int) if faces is not None and len(faces) > 0 else np.empty((0, 3), dtype=int)
        self.fixed_nodes = set(fixed_nodes)
        self.perimeter_nodes = set(perimeter_nodes)
        self.material_type = str(material_type).lower()

        # Domain extents
        if len(self.nodes) > 0:
            self.xmin, self.ymin, _ = np.min(self.nodes, axis=0)
            self.xmax, self.ymax, _ = np.max(self.nodes, axis=0)
        else:
            self.xmin, self.ymin = 0.0, 0.0
            self.xmax, self.ymax = Lx, Ly

        self.Lx = float(self.xmax - self.xmin)
        self.Ly = float(self.ymax - self.ymin)
        self.Lz = float(Lz)

        # Average spacing for downstream width metrics
        if len(self.edges) > 0:
            edge_lens = np.linalg.norm(self.nodes[self.edges[:, 0]] - self.nodes[self.edges[:, 1]], axis=1)
            mean_len = float(np.mean(edge_lens))
            self.dx = mean_len
            self.dy = mean_len
        else:
            self.dx, self.dy = 100.0, 100.0

    @staticmethod
    def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Vectorized ray-casting Point-in-Polygon algorithm for 2D points."""
        x, y = points[:, 0], points[:, 1]
        n_poly = len(polygon)
        inside = np.zeros(len(points), dtype=bool)

        p1x, p1y = polygon[0]
        for i in range(n_poly + 1):
            p2x, p2y = polygon[i % n_poly]
            idx = np.where((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)))[0]
            if len(idx) > 0:
                if p1y != p2y:
                    xinters = (y[idx] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    inside[idx] ^= (p1x == p2x) | (x[idx] <= xinters)
            p1x, p1y = p2x, p2y

        return inside

    @classmethod
    def build_polygon_domain(
        cls,
        perimeter_pts: list,
        interior_pts: list = None,
        line_supports: list = None,
        target_edge_len: float = 250.0,
        material_type: str = "membrane"
    ):
        """
        Builds an unstructured form-finding mesh for arbitrary planar polygon domains.
        Enforces boundary segments, interior point supports, and linear support paths.
        """
        poly_2d = np.array([[float(pt["x"]), float(pt["y"])] for pt in perimeter_pts], dtype=float)
        n_perim = len(poly_2d)

        if n_perim < 3:
            raise ValueError("Perimeter polygon requires at least 3 ordered points.")

        # 1. Base Explicit Sites
        nodes_list = []
        fixed_indices = set()
        perimeter_indices = set()
        curr_idx = 0

        # Ingest Ordered Perimeter Nodes
        for pt in perimeter_pts:
            nodes_list.append([float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0))])
            perimeter_indices.add(curr_idx)
            fixed_indices.add(curr_idx)
            curr_idx += 1

        # Ingest Interior Point Supports
        if interior_pts:
            for pt in interior_pts:
                nodes_list.append([float(pt["x"]), float(pt["y"]), float(pt.get("z", 0.0))])
                fixed_indices.add(curr_idx)
                curr_idx += 1

        # Ingest Discretized Line Support Points
        if line_supports:
            for lsup in line_supports:
                p1 = np.array([float(lsup["x1"]), float(lsup["y1"]), float(lsup.get("z1", 0.0))])
                p2 = np.array([float(lsup["x2"]), float(lsup["y2"]), float(lsup.get("z2", 0.0))])
                length = np.linalg.norm(p2[:2] - p1[:2])
                n_seg = max(int(np.ceil(length / target_edge_len)), 1)
                for step in range(n_seg + 1):
                    t = step / n_seg
                    pt_interp = p1 + t * (p2 - p1)
                    nodes_list.append([pt_interp[0], pt_interp[1], pt_interp[2]])
                    fixed_indices.add(curr_idx)
                    curr_idx += 1

        # 2. Fill Background Free Grid within Polygon
        min_x, min_y = np.min(poly_2d, axis=0)
        max_x, max_y = np.max(poly_2d, axis=0)
        gx = np.arange(min_x, max_x, target_edge_len)
        gy = np.arange(min_y, max_y, target_edge_len)
        grid_x, grid_y = np.meshgrid(gx, gy)
        candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        inside_mask = cls._point_in_polygon(candidates, poly_2d)
        internal_grid = candidates[inside_mask]

        explicit_sites = np.array([n[:2] for n in nodes_list], dtype=float)
        if len(internal_grid) > 0 and len(explicit_sites) > 0:
            dists = np.linalg.norm(internal_grid[:, None, :] - explicit_sites[None, :, :], axis=2)
            valid_grid = internal_grid[np.min(dists, axis=1) > (target_edge_len * 0.45)]
            for pt in valid_grid:
                nodes_list.append([pt[0], pt[1], 0.0])
                curr_idx += 1

        nodes = np.array(nodes_list, dtype=float)

        # 3. Triangulation Pass & Centroid Filtering
        tri = Delaunay(nodes[:, :2])
        centroids = np.mean(nodes[tri.simplices, :2], axis=1)
        valid_mask = cls._point_in_polygon(centroids, poly_2d)
        valid_faces = tri.simplices[valid_mask]

        # 4. Boundary Edge Recovery Pass
        # Verify that all perimeter segments (0->1, 1->2, ..., n-1->0) exist in graph
        raw_edges = set()
        for face in valid_faces:
            for i in range(3):
                u, v = face[i], face[(i + 1) % 3]
                raw_edges.add(tuple(sorted((u, v))))

        for i in range(n_perim):
            u = i
            v = (i + 1) % n_perim
            edge_tuple = tuple(sorted((u, v)))
            if edge_tuple not in raw_edges:
                # Force edge connection for perimeter integrity
                raw_edges.add(edge_tuple)

        edges = np.array(list(raw_edges), dtype=int)
        Lx_calc = max_x - min_x
        Ly_calc = max_y - min_y
        Lz_calc = float(np.max(nodes[:, 2])) if len(nodes) > 0 else 1000.0

        return cls(
            nodes=nodes,
            edges=edges,
            faces=valid_faces,
            fixed_nodes=fixed_indices,
            perimeter_nodes=perimeter_indices,
            material_type=material_type,
            Lx=Lx_calc,
            Ly=Ly_calc,
            Lz=Lz_calc
        )

    def apply_idw_surface_interpolation(self, seed_z_map: dict, power: float = 2.0):
        if self.material_type in ("cables", "cable") or len(self.nodes) == 0 or not seed_z_map:
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
