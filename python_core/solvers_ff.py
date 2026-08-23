# DBSW 3D Dynamic Relaxation & Dynamic Form-Finding Solver
# Author: Damian Brenlla / DBSW 2026

import numpy as np


class UniversalFormFindingSolver:
    """Dynamic Relaxation solver for 3D form-finding (vaults, domes, cable nets, fabrics).

    Iteratively updates nodal positions until internal member forces equilibrate
    external point loads and gravity self-weight.
    """

    def __init__(
        self,
        domain,
        E_modulus: float = 210000.0,
        gamma_kn_m3: float = 78.5,
        cross_section_area: float = 90000.0,
        area_mm2: float = None,
        prestress_force: float = 15.0,
        point_loads: list = None,
    ):
        self.domain = domain
        self.E = float(E_modulus)
        self.gamma_kn_m3 = float(gamma_kn_m3)
        
        # Support both 'cross_section_area' and 'area_mm2' keyword arguments
        if area_mm2 is not None:
            self.area = float(area_mm2)
        else:
            self.area = float(cross_section_area)

        self.prestress = float(prestress_force)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations: int = 300, invert_form: bool = False):
        """Runs dynamic relaxation to find nodal positions at static equilibrium."""
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        # Calculate initial member lengths
        rest_lengths = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            dist = np.linalg.norm(nodes[u] - nodes[v])
            rest_lengths[i] = dist if dist > 1e-6 else 1.0

        # Nodal external force vectors (N)
        F_ext = np.zeros((num_nodes, 3), dtype=float)

        # 1. Apply gravity self-weight if gamma > 0
        if self.gamma_kn_m3 > 0.0:
            density_kg_mm3 = (self.gamma_kn_m3 / 9.81) * 1e-9
            for i, (u, v) in enumerate(edges):
                L = rest_lengths[i]
                member_mass_kg = density_kg_mm3 * self.area * L
                half_weight_N = (member_mass_kg * 9.81) / 2.0
                F_ext[u, 2] -= half_weight_N
                F_ext[v, 2] -= half_weight_N

        # 2. Map discrete spatial point loads (kN -> N) onto closest grid nodes
        for ld in self.point_loads:
            px, py, pz = float(ld.get("x", 0)), float(ld.get("y", 0)), float(ld.get("z", 0))
            fx_N = float(ld.get("Fx", 0)) * 1000.0
            fy_N = float(ld.get("Fy", 0)) * 1000.0
            fz_N = float(ld.get("Fz", 0)) * 1000.0

            # Find nearest node
            dists = np.linalg.norm(nodes - np.array([px, py, pz]), axis=1)
            closest_idx = int(np.argmin(dists))
            F_ext[closest_idx, 0] += fx_N
            F_ext[closest_idx, 1] += fy_N
            F_ext[closest_idx, 2] += fz_N

        # Invert Z forces for funicular compression forms (Vaults/Domes)
        if invert_form:
            F_ext[:, 2] = -F_ext[:, 2]

        # Dynamic Relaxation Iteration Loop
        velocities = np.zeros((num_nodes, 3), dtype=float)
        damping = 0.85
        dt = 0.01

        axial_forces = np.zeros(num_edges, dtype=float)
        reactions = np.zeros((num_nodes, 3), dtype=float)

        for _ in range(iterations):
            F_int = np.zeros((num_nodes, 3), dtype=float)

            for i, (u, v) in enumerate(edges):
                vec = nodes[v] - nodes[u]
                curr_len = np.linalg.norm(vec)
                if curr_len < 1e-6:
                    continue
                unit_vec = vec / curr_len

                # Axial strain + prestress axial force (N)
                strain = (curr_len - rest_lengths[i]) / rest_lengths[i]
                force = (self.E * self.area * strain) + self.prestress
                axial_forces[i] = force

                f_vec = force * unit_vec
                F_int[u] += f_vec
                F_int[v] -= f_vec

            # Update nodal motion
            for i in range(num_nodes):
                if i in fixed_nodes:
                    # Support reaction force recovery (N)
                    reactions[i] = -(F_int[i] + F_ext[i])
                    velocities[i] = 0.0
                else:
                    R_residual = F_ext[i] + F_int[i]
                    velocities[i] = (velocities[i] + R_residual * dt) * damping
                    nodes[i] += velocities[i] * dt

        # Flip geometry back for compression funicular arches/vaults
        if invert_form:
            max_z = np.max(nodes[:, 2])
            nodes[:, 2] = max_z - nodes[:, 2]
            reactions[:, 2] = -reactions[:, 2]

        return nodes, axial_forces, reactions
