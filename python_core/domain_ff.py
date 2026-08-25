# DBSW Spatial Form-Finding Network Domain
# Author: Damian Brenlla / DBSW 2026
# v3 — Added 3D vector line supports (arbitrary spatial vectors between 2 points)

import numpy as np


class FormFindingDomain3D:
    """
    Node-Edge Spatial Network for Vaults, Domes, Cable Nets, and Gridshells.

    All geometry presets start FLAT (Z=0) so the dynamic relaxation solver 
    finds the true hanging funicular shape under gravity. The inversion 
    to a compression vault happens after solving, not before.
    """

    def __init__(
        self,
        Lx: float,
        Ly: float,
        Lz: float,
        nx: int,
        ny: int,
        geometry_preset: str = "surface_grid",
    ):
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.nx = int(max(nx, 2))
        self.ny = int(max(ny, 2))
        self.geometry_preset = geometry_preset

        self.nodes = []
        self.edges = []
        self.fixed_nodes = set()
        self.node_loads = {}

        self._build_network_topology()

    def _build_network_topology(self):
        """
        Generates flat (Z=0) node-edge connectivity.
        Starting flat is essential for correct funicular form-finding.
        The solver applies loads and finds the equilibrium shape from scratch.
        """
        x_lin = np.linspace(0, self.Lx, self.nx + 1)
        y_lin = np.linspace(0, self.Ly, self.ny + 1)
        grid_map = {}
        node_idx = 0

        for i, x in enumerate(x_lin):
            for j, y in enumerate(y_lin):
                # Always start flat — let the solver find the shape
                self.nodes.append([x, y, 0.0])
                grid_map[(i, j)] = node_idx
                node_idx += 1

        # Grid edges — longitudinal, transverse, and diagonal bracing
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                curr = grid_map[(i, j)]

                # Longitudinal (X-direction)
                if i < self.nx:
                    self.edges.append([curr, grid_map[(i + 1, j)]])

                # Transverse (Y-direction)
                if j < self.ny:
                    self.edges.append([curr, grid_map[(i, j + 1)]])

                # Diagonal bracing — both diagonals for stability
                if i < self.nx and j < self.ny:
                    self.edges.append([curr, grid_map[(i + 1, j + 1)]])
                    self.edges.append([grid_map[(i + 1, j)], grid_map[(i, j + 1)]])

        self.nodes = np.array(self.nodes, dtype=float)
        self.edges = np.array(self.edges, dtype=int)

    def _auto_tolerance(self) -> float:
        """
        Calculates node-proximity tolerance from grid spacing.
        Ensures support/load snapping works regardless of mesh density.
        """
        dx = self.Lx / self.nx if self.nx > 0 else 100.0
        dy = self.Ly / self.ny if self.ny > 0 else 100.0
        return max(dx, dy) * 0.55

    def add_point_support(self, x: float, y: float, z: float, tol: float = None):
        """Fixes the closest node to (x, y, z) within tolerance."""
        if tol is None:
            tol = self._auto_tolerance()
        dists = np.linalg.norm(self.nodes - np.array([x, y, z]), axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= tol:
            self.fixed_nodes.add(idx)

    def add_line_support(self, axis: str = "x", value: float = 0.0, tol: float = None):
        """
        Fixes all nodes along an axis alignment at the given coordinate.
        """
        if tol is None:
            tol = self._auto_tolerance()

        axis_map = {"x": 0, "y": 1, "z": 2}
        col = axis_map[axis.lower()]
        matches = np.where(np.abs(self.nodes[:, col] - value) <= tol)[0]
        for idx in matches:
            self.fixed_nodes.add(int(idx))

    def add_line_support_3d(self, p1: tuple, p2: tuple, tol: float = None):
        """
        Fixes all nodes lying along a 3D line segment vector between p1 and p2.
        p1: (x1, y1, z1)
        p2: (x2, y2, z2)
        """
        if tol is None:
            tol = self._auto_tolerance()

        p1_arr = np.array(p1, dtype=float)
        p2_arr = np.array(p2, dtype=float)
        line_vec = p2_arr - p1_arr
        line_len = np.linalg.norm(line_vec)

        # If length is tiny, treat as a single point support
        if line_len < 1e-3:
            self.add_point_support(p1_arr[0], p1_arr[1], p1_arr[2], tol)
            return

        line_dir = line_vec / line_len

        for idx, node in enumerate(self.nodes):
            node_vec = node - p1_arr
            proj_len = np.dot(node_vec, line_dir)

            # Check if node projection falls within segment bounds [0, line_len]
            if -tol <= proj_len <= line_len + tol:
                proj_pt = p1_arr + np.clip(proj_len, 0.0, line_len) * line_dir
                dist = np.linalg.norm(node - proj_pt)
                if dist <= tol:
                    self.fixed_nodes.add(idx)

    def add_edge_support(self, edge: str = "all"):
        """
        Fixes nodes along named boundary edges.
        Options: 'all', 'x0', 'xmax', 'y0', 'ymax'
        """
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
        """Returns indices of all perimeter nodes."""
        tol = self._auto_tolerance()
        mask = (
            (np.abs(self.nodes[:, 0]) < tol) |
            (np.abs(self.nodes[:, 0] - self.Lx) < tol) |
            (np.abs(self.nodes[:, 1]) < tol) |
            (np.abs(self.nodes[:, 1] - self.Ly) < tol)
        )
        return np.where(mask)[0]

    def get_stats(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_fixed": len(self.fixed_nodes),
            "grid_spacing_x": self.Lx / self.nx if self.nx > 0 else 0,
            "grid_spacing_y": self.Ly / self.ny if self.ny > 0 else 0,
        }
