# DBSW 3D Dynamic Relaxation & Dynamic Form-Finding Solver
# Author: Damian Brenlla / DBSW 2026

import numpy as np


class UniversalFormFindingSolver:
    """Robust Dynamic Relaxation solver for 3D form-finding (vaults, domes, cable nets, fabrics)."""

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
        self.E = max(float(E_modulus), 1.0)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)

        if area_mm2 is not None:
            self.area = max(float(area_mm2), 1e-4)
        else:
            self.area = max(float(cross_section_area), 1e-4)

        self.prestress = float(prestress_force)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations: int = 300, invert_form: bool = False):
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        if num_nodes == 0 or num_edges == 0:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3))

        rest_lengths = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            dist = np.linalg.norm(nodes[u] - nodes[v])
            rest_lengths[i] = dist if dist > 1e-4 else 1.0

        F_ext = np.zeros((num_nodes, 3), dtype=float)

        # 1. Apply gravity self-weight
        if self.gamma_kn_m3 > 0.0:
            density_kg_mm3 = (self.gamma_kn_m3 / 9.81) * 1e-9
            for i, (u, v) in enumerate(edges):
                L = rest_lengths[i]
                member_mass_kg = density_kg_mm3 * self.area * L
                half_weight_N = (member_mass_kg * 9.81) / 2.0
                F_ext[u, 2] -= half_weight_N
                F_ext[v, 2] -= half_weight_N

        # 2. Map discrete point loads onto closest grid nodes
        for ld in self.point_loads:
            px, py, pz = float(ld.get("x", 0)), float(ld.get("y", 0)), float(ld.get("z", 0))
            fx_N = float(ld.get("Fx", 0)) * 1000.0
            fy_N = float(ld.get("Fy", 0)) * 1000.0
            fz_N = float(ld.get("Fz", 0)) * 1000.0

            dists = np.linalg.norm(nodes - np.array([px, py, pz]), axis=1)
            closest_idx = int(np.argmin(dists))
            F_ext[closest_idx, 0] += fx_N
            F_ext[closest_idx, 1] += fy_N
            F_ext[closest_idx, 2] += fz_N

        if invert_form:
            F_ext[:, 2] = -F_ext[:, 2]

        velocities = np.zeros((num_nodes, 3), dtype=float)
        damping = 0.85
        dt = 0.005

        axial_forces = np.zeros(num_edges, dtype=float)
        reactions = np.zeros((num_nodes, 3), dtype=float)

        for _ in range(max(iterations, 10)):
            F_int = np.zeros((num_nodes, 3), dtype=float)

            for i, (u, v) in enumerate(edges):
                vec = nodes[v] - nodes[u]
                curr_len = np.linalg.norm(vec)
                if curr_len < 1e-6:
                    continue
                unit_vec = vec / curr_len

                L0 = rest_lengths[i]
                strain = (curr_len - L0) / L0
                force = (self.E * self.area * strain) + self.prestress
                
                if np.isnan(force) or np.isinf(force):
                    force = 0.0
                axial_forces[i] = force

                f_vec = force * unit_vec
                F_int[u] += f_vec
                F_int[v] -= f_vec

            for i in range(num_nodes):
                if i in fixed_nodes:
                    reactions[i] = -(F_int[i] + F_ext[i])
                    velocities[i] = 0.0
                else:
                    R_residual = F_ext[i] + F_int[i]
                    velocities[i] = (velocities[i] + R_residual * dt) * damping
                    nodes[i] += velocities[i] * dt

        if invert_form:
            max_z = np.max(nodes[:, 2])
            nodes[:, 2] = max_z - nodes[:, 2]
            reactions[:, 2] = -reactions[:, 2]

        # Final NaN/Inf Sanitization Safeguard
        nodes = np.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)
        axial_forces = np.nan_to_num(axial_forces, nan=0.0, posinf=0.0, neginf=0.0)
        reactions = np.nan_to_num(reactions, nan=0.0, posinf=0.0, neginf=0.0)

        return nodes, axial_forces, reactions
