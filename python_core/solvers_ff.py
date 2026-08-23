# DBSW Universal Dynamic Relaxation & Force Density Solver Engine
# Author: Damian Brenlla / DBSW 2026

import numpy as np


class UniversalFormFindingSolver:
    """Dynamic Relaxation (DR) & Force Density Method (FDM) Solver."""

    def __init__(self, domain, E_modulus: float, gamma_kn_m3: float, prestress_force: float = 10.0):
        self.domain = domain
        self.E = E_modulus
        self.gamma_n_mm3 = gamma_kn_m3 * 1e-6
        self.prestress = prestress_force  # N

    def solve_equilibrium(self, iterations: int = 300, invert_form: bool = False, damping: float = 0.85):
        nodes = np.copy(self.domain.nodes)
        num_nodes = len(nodes)

        fixed_indices = list(self.domain.fixed_nodes)
        free_indices = np.setdiff1d(np.arange(num_nodes), fixed_indices)

        velocities = np.zeros_like(nodes)
        forces = np.zeros_like(nodes)
        reactions = np.zeros_like(nodes)

        node_masses = np.ones(num_nodes) * 0.1  # kg
        t_dir = 1.0 if invert_form else -1.0

        for _ in range(iterations):
            forces.fill(0.0)

            # 1. Gravity / external forces
            forces[:, 2] += t_dir * node_masses * 9.81

            # 2. Compute element forces
            node_a = nodes[self.domain.edges[:, 0]]
            node_b = nodes[self.domain.edges[:, 1]]
            vecs = node_b - node_a
            lengths = np.linalg.norm(vecs, axis=1)
            lengths[lengths == 0] = 1e-6

            dirs = vecs / lengths[:, np.newaxis]
            axial_forces = self.prestress * (lengths / np.mean(lengths))

            np.add.at(forces, self.domain.edges[:, 0], dirs * axial_forces[:, np.newaxis])
            np.add.at(forces, self.domain.edges[:, 1], -dirs * axial_forces[:, np.newaxis])

            # 3. Dynamic Relaxation Integration
            velocities[free_indices] = (velocities[free_indices] * damping) + (forces[free_indices] / node_masses[free_indices][:, np.newaxis])
            nodes[free_indices] += velocities[free_indices]

            if np.sum(velocities * forces) < 0:
                velocities.fill(0.0)

        # 4. Support Reactions
        reactions[fixed_indices] = -forces[fixed_indices]

        if invert_form:
            nodes[:, 2] = np.max(nodes[:, 2]) - nodes[:, 2]

        return nodes, axial_forces, reactions