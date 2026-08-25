# DBSW Spatial Form-Finding Network Domain
# Author: Damian Brenlla / DBSW 2026
# v12 — Fixed: fixed-node protection during bilinear seeding, line-support coordinate
#       preservation for already-fixed nodes, tightened boundary tolerance, dx/dy exposure.

import numpy as np


class FormFindingDomain3D:
    def __init__(
        self,
        Lx: float,
        Ly: float,
        Lz: float,
        nx: int,
        ny: int,
        geometry_preset: str = "surface_grid",
        material_type: str = "cables"
    ):
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.nx = int(max(nx, 2))
        self.ny = int(max(ny, 2))
        self.geometry_preset = geometry_preset
        self.material_type = str(material_type).lower()

        self.nodes = []
        self.edges = []
        self.fixed_nodes = set()
        self.node_loads = {}

        self._build_network_topology()

        # Expose grid spacing so solvers see real membrane strip widths
        self.dx = self.Lx / self.nx if self.nx > 0 else 0.0
        self.dy = self.Ly / self.ny if self.ny > 0 else 0.0

    def _build_network_topology(self):
        is_pure_cable = self.material_type in ("cables", "cable")

        if is_pure_cable:
            x_lin = np.linspace(0, self.Lx, self.nx + 1)
            y_lin = np.linspace(0, self.Ly if self.Ly > 0 else 0, self.nx + 1)
            for i in range(self.nx + 1):
                self.nodes.append([x_lin[i], y_lin[i], 0.0])
                if i < self.nx:
                    self.edges.append([i, i + 1])
        else:
            x_lin = np.linspace(0, self.Lx, self.nx + 1)
            y_lin = np.linspace(0, self.Ly, self.ny + 1)
            grid_map = {}
            node_idx = 0
            for i, x in enumerate(x_lin):
                for j, y in enumerate(y_lin):
                    self.nodes.append([x, y, 0.0])
                    grid_map[(i, j)] = node_idx
                    node_idx += 1

            for i in range(self.nx + 1):
                for j in range(self.ny + 1):
                    curr = grid_map[(i, j)]
                    if i < self.nx:
                        self.edges.append([curr, grid_map[(i + 1, j)]])
                    if j < self.ny:
                        self.edges.append([curr, grid_map[(i, j + 1)]])
                    if i < self.nx and j < self.ny:
                        self.edges.append([curr, grid_map[(i + 1, j + 1)]])
                        self.edges.append([grid_map[(i + 1, j)], grid_map[(i, j + 1)]])

        self.nodes = np.array(self.nodes, dtype=float)
        self.edges = np.array(self.edges, dtype=int)

    def _auto_tolerance(self) -> float:
        dx = self.Lx / self.nx if self.nx > 0 else 100.0
        dy = self.Ly / self.ny if self.ny > 0 else 100.0
        return max(dx, dy) * 1.5

    def apply_bilinear_surface_interpolation(self, corner_z_map: dict):
        """
        Smooths initial 2D surface grid Z-elevations using 3D bilinear corner interpolation.
        STRICT GUARDRAIL: Only executes for 2D surface manifolds, never 1D cables.
        CRITICAL FIX: Preserves Z of nodes already locked by point supports.
        """
        if self.material_type in ("cables", "cable") or len(self.nodes) == 0:
            return

        z00 = 0.0
        z10 = 0.0
        z01 = 0.0
        z11 = 0.0

        for (pt_x, pt_y), pt_z in corner_z_map.items():
            if pt_x <= self.Lx * 0.25 and pt_y <= self.Ly * 0.25:
                z00 = float(pt_z)
            elif pt_x >= self.Lx * 0.75 and pt_y <= self.Ly * 0.25:
                z10 = float(pt_z)
            elif pt_x <= self.Lx * 0.25 and pt_y >= self.Ly * 0.75:
                z01 = float(pt_z)
            elif pt_x >= self.Lx * 0.75 and pt_y >= self.Ly * 0.75:
                z11 = float(pt_z)

        xi = self.nodes[:, 0] / max(self.Lx, 1e-3)
        eta = self.nodes[:, 1] / max(self.Ly, 1e-3)

        interpolated_z = (
            (1.0 - xi) * (1.0 - eta) * z00 +
            xi * (1.0 - eta) * z10 +
            (1.0 - xi) * eta * z01 +
            xi * eta * z11
        )

        # CRITICAL FIX: Do not overwrite Z of nodes already fixed by point supports
        free_mask = np.ones(len(self.nodes), dtype=bool)
        if self.fixed_nodes:
            free_mask[list(self.fixed_nodes)] = False
        self.nodes[free_mask, 2] = interpolated_z[free_mask]

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
        CRITICAL FIX: Already-fixed point-support nodes are NOT moved to the line.
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

        # CRITICAL FIX: Tightened candidate pool to outermost row only
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
                    # CRITICAL FIX: Only snap position if node wasn't already fixed by a point support
                    if not already_fixed:
                        self.nodes[idx] = proj_pt

    def add_edge_support(self, edge: str = "all"):
        tol = self._auto_tolerance()
        if edge in ("all", "x0"):
            self.add_line_support("x", 0.0, tol)
        if edge in ("all", "xmax"):
            self.add_line_support("x", self.Lx, tol)
        if edge in ("all", "y0"):
            self.add_line_support("y", 0.0, tol)
        if edge in ("all", "ymax"):
            self.add_line_support("y", self.Ly, tol)

    def get_boundary_nodes(self) -> np.ndarray:
        # CRITICAL FIX: Tightened tolerance so only the outermost row of nodes is captured
        dx = self.Lx / self.nx if self.nx > 0 else 100.0
        dy = self.Ly / self.ny if self.ny > 0 else 100.0
        tol = max(dx, dy) * 0.51
        mask = (
            (np.abs(self.nodes[:, 0]) < tol)
            | (np.abs(self.nodes[:, 0] - self.Lx) < tol)
            | (np.abs(self.nodes[:, 1]) < tol)
            | (np.abs(self.nodes[:, 1] - self.Ly) < tol)
        )
        return np.where(mask)[0]

    def get_stats(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_fixed": len(self.fixed_nodes),
            "grid_spacing_x": self.dx,
            "grid_spacing_y": self.dy,
        }
