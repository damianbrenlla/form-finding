# DBSW Spatial Form-Finding Network Domain
# Author: Damian Brenlla / DBSW 2026

import numpy as np


class FormFindingDomain3D:
    """Node-Edge Spatial Network for Vaults, Domes, Cables, and Gridshells."""

    def __init__(self, Lx: float, Ly: float, Lz: float, nx: int, ny: int, geometry_preset: str = "surface_grid"):
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.nx = int(nx)
        self.ny = int(ny)
        self.geometry_preset = geometry_preset

        self.nodes = []
        self.edges = []
        self.fixed_nodes = set()
        self.node_loads = {}

        self._build_network_topology()

    def _build_network_topology(self):
        """Generates topological node-edge connectivity based on selected geometry preset."""
        x_lin = np.linspace(0, self.Lx, self.nx + 1)
        y_lin = np.linspace(0, self.Ly, self.ny + 1)
        grid_map = {}

        # Generate Nodes
        node_idx = 0
        for i, x in enumerate(x_lin):
            for j, y in enumerate(y_lin):
                z = 0.0
                if self.geometry_preset in ["vault", "catenary_arch"]:
                    z = self.Lz * np.sin(np.pi * x / self.Lx)
                elif self.geometry_preset == "dome":
                    cx, cy = self.Lx / 2.0, self.Ly / 2.0
                    r_norm = np.sqrt(((x - cx) / cx)**2 + ((y - cy) / cy)**2)
                    z = max(0.0, self.Lz * (1.0 - r_norm**2))

                self.nodes.append([x, y, z])
                grid_map[(i, j)] = node_idx
                node_idx += 1

        # Generate Edges
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                curr = grid_map[(i, j)]
                if i < self.nx:
                    self.edges.append([curr, grid_map[(i + 1, j)]])
                if j < self.ny:
                    self.edges.append([curr, grid_map[(i, j + 1)]])
                # Diagonal bracing for shear/fabric stability
                if i < self.nx and j < self.ny:
                    self.edges.append([curr, grid_map[(i + 1, j + 1)]])

        self.nodes = np.array(self.nodes, dtype=float)
        self.edges = np.array(self.edges, dtype=int)

    def add_point_support(self, x: float, y: float, z: float, tol: float = 50.0):
        dists = np.linalg.norm(self.nodes - np.array([x, y, z]), axis=1)
        idx = np.argmin(dists)
        if dists[idx] <= tol:
            self.fixed_nodes.add(idx)

    def add_line_support(self, axis: str = "x", value: float = 0.0, tol: float = 10.0):
        axis_map = {"x": 0, "y": 1, "z": 2}
        col = axis_map[axis.lower()]
        matches = np.where(np.abs(self.nodes[:, col] - value) <= tol)[0]
        for idx in matches:
            self.fixed_nodes.add(idx)