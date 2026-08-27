# DBSW Spatial Form-Finding Network Domain
# Author: Damian Brenlla / DBSW 2026
# v13 — Arbitrary bounding box (negative coordinates now inside the domain), forced
#       grid lines at every support coordinate so a support always lands on a real
#       node (add_point_support now fails LOUDLY instead of silently doing nothing),
#       and inverse-distance-weighted seeding across ALL supports in place of the
#       old 4-corner-only bilinear bucket classifier.

import numpy as np


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
        material_type: str = "cables",
        build_topology: bool = True
    ):
        self.xmin = float(xmin)
        self.xmax = float(xmax) if xmax > xmin else float(xmin) + 1.0
        self.ymin = float(ymin)
        self.ymax = float(ymax) if ymax > ymin else float(ymin) + 1.0

        # Kept for backwards compatibility with any code (incl. solvers) that reads
        # domain.Lx / domain.Ly as the SPAN of the domain — never assume origin is 0.
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.Lz = float(Lz)
        self.nx = int(max(nx, 2))
        self.ny = int(max(ny, 2))
        self.geometry_preset = geometry_preset
        self.material_type = str(material_type).lower()

        self.nodes = []
        self.edges = []
        self.fixed_nodes = set()
        self.node_loads = {}
        # Triangle connectivity (filled for irregular/polygon meshes; None for
        # the legacy structured rectangular grid, where the frontend rebuilds
        # triangles from the nx/ny stride).
        self.triangles = None

        self._x_lin = None
        self._y_lin = None

        if build_topology:
            self._build_network_topology(forced_x or [], forced_y or [])

            # Average spacing (grid may now be non-uniform due to forced lines near
            # supports) — used for tolerance checks and prestress-per-metre widths.
            self.dx = float(np.mean(np.diff(self._x_lin))) if len(self._x_lin) > 1 else 0.0
            self.dy = float(np.mean(np.diff(self._y_lin))) if len(self._y_lin) > 1 else 0.0
        else:
            # Pre-built mesh path: nodes/edges/fixed_nodes/triangles are injected
            # directly by the caller (e.g. an irregular polygon Delaunay mesh built
            # in the worker). Provide sane dx/dy fallbacks so solvers that read
            # domain.dx/dy for prestress-per-metre widths don't divide by zero.
            self.dx = self.Lx / max(self.nx, 1)
            self.dy = self.Ly / max(self.ny, 1)
            self.nodes = np.zeros((0, 3), dtype=float)
            self.edges = np.zeros((0, 2), dtype=int)

    def _build_network_topology(self, forced_x, forced_y):
        is_pure_cable = self.material_type in ("cables", "cable")

        if is_pure_cable:
            x_lin = np.linspace(self.xmin, self.xmax, self.nx + 1)
            y_lin = np.linspace(self.ymin, self.ymax, self.nx + 1)
            self._x_lin, self._y_lin = x_lin, y_lin
            for i in range(self.nx + 1):
                self.nodes.append([x_lin[i], y_lin[i], 0.0])
                if i < self.nx:
                    self.edges.append([i, i + 1])
        else:
            x_lin = np.linspace(self.xmin, self.xmax, self.nx + 1)
            y_lin = np.linspace(self.ymin, self.ymax, self.ny + 1)

            # CRITICAL FIX: force every support coordinate to be an exact grid line
            # instead of hoping the nearest sampled line falls within tolerance.
            # np.union1d also sorts + dedups, so passing raw (possibly repeated)
            # support coordinates straight through is safe.
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
            for i in range(nx_actual + 1):
                for j in range(ny_actual + 1):
                    curr = grid_map[(i, j)]
                    if i < nx_actual:
                        self.edges.append([curr, grid_map[(i + 1, j)]])
                    if j < ny_actual:
                        self.edges.append([curr, grid_map[(i, j + 1)]])
                    if i < nx_actual and j < ny_actual:
                        self.edges.append([curr, grid_map[(i + 1, j + 1)]])
                        self.edges.append([grid_map[(i + 1, j)], grid_map[(i, j + 1)]])

            # Forced lines mean actual resolution may exceed the requested nx/ny —
            # keep these in sync so downstream tolerance math stays correct.
            self.nx = nx_actual
            self.ny = ny_actual

        self.nodes = np.array(self.nodes, dtype=float)
        self.edges = np.array(self.edges, dtype=int)

    def _auto_tolerance(self) -> float:
        dx = self.Lx / self.nx if self.nx > 0 else 100.0
        dy = self.Ly / self.ny if self.ny > 0 else 100.0
        # Forced grid lines put a real node at every support coordinate, so this
        # only needs to cover floating-point slop — not "nearest sampled line".
        return max(dx, dy) * 0.51

    def apply_idw_surface_interpolation(self, seed_z_map: dict, power: float = 2.0):
        """
        Seeds initial Z-elevations for all FREE nodes using inverse-distance
        weighting against every declared support point — not just 4 corners —
        so line supports, off-corner points, and negative-coordinate supports
        all contribute correctly to the starting shape.
        STRICT GUARDRAIL: only runs for 2D surface manifolds, never 1D cables.
        Never overwrites Z of nodes already locked by point/line supports.
        """
        if self.material_type in ("cables", "cable") or len(self.nodes) == 0:
            return
        if not seed_z_map:
            return

        pts = np.array(list(seed_z_map.keys()), dtype=float)      # (P, 2)
        zvals = np.array(list(seed_z_map.values()), dtype=float)  # (P,)

        free_mask = np.ones(len(self.nodes), dtype=bool)
        if self.fixed_nodes:
            free_mask[list(self.fixed_nodes)] = False

        free_xy = self.nodes[free_mask, :2]
        if len(free_xy) == 0:
            return

        # Distance from every free node to every seed point: (N_free, P)
        diffs = free_xy[:, None, :] - pts[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        weights = 1.0 / np.maximum(dists, 1e-6) ** power
        z_interp = (weights @ zvals) / np.sum(weights, axis=1)

        # A free node that happens to sit exactly on a seed point (distance ~0)
        # takes that seed's Z directly rather than the IDW blend.
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

            if self.material_type in ("cables", "cable") and len(self.fixed_nodes) >= 2:
                fixed_list = sorted(list(self.fixed_nodes))
                p1_idx, p2_idx = fixed_list[0], fixed_list[-1]
                pos1 = self.nodes[p1_idx].copy()
                pos2 = self.nodes[p2_idx].copy()
                for axis in range(3):
                    self.nodes[:, axis] = np.linspace(pos1[axis], pos2[axis], len(self.nodes))
            return idx

        # CRITICAL FIX: fail loudly instead of silently doing nothing. With forced
        # grid lines in place (see worker_ff.js) this should essentially never
        # trigger — if it does, something upstream failed to register the support
        # coordinate as a forced grid line, and that's a bug worth surfacing.
        raise ValueError(
            f"Point support at ({x}, {y}) is {dists_2d[idx]:.2f}mm from the nearest "
            f"mesh node — outside tolerance ({tol:.2f}mm). It was NOT attached, which "
            f"is why the fabric was dropping at this location."
        )

    def add_line_support(self, axis: str = "x", value: float = 0.0, tol: float = None):
        if tol is None:
            tol = self._auto_tolerance()

        axis_map = {"x": 0, "y": 1, "z": 2}
        col = axis_map[axis.lower()]
        matches = np.where(np.abs(self.nodes[:, col] - value) <= tol)[0]
        for idx in matches:
            self.fixed_nodes.add(int(idx))

    def add_line_support_3d(self, p1: tuple, p2: tuple, tol: float = None):
        """
        Constrains boundary nodes along a 3D line support vector.
        Already-fixed point-support nodes are NOT moved to the line.
        """
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

        candidate_indices = (
            self.get_boundary_nodes()
            if self.material_type not in ("cables", "cable")
            else range(len(self.nodes))
        )

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
        if edge in ("all", "x0"):
            self.add_line_support("x", self.xmin, tol)
        if edge in ("all", "xmax"):
            self.add_line_support("x", self.xmax, tol)
        if edge in ("all", "y0"):
            self.add_line_support("y", self.ymin, tol)
        if edge in ("all", "ymax"):
            self.add_line_support("y", self.ymax, tol)

    def get_boundary_nodes(self) -> np.ndarray:
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
            "num_fixed": len(self.fixed_nodes),
            "grid_spacing_x": self.dx,
            "grid_spacing_y": self.dy,
            "xmin": self.xmin, "xmax": self.xmax,
            "ymin": self.ymin, "ymax": self.ymax,
        }
