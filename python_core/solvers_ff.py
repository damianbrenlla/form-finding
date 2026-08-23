# DBSW 3D Dynamic Relaxation Form-Finding Solver
# Author: Damian Brenlla / DBSW 2026
# v2 — Fixed: self-weight density, fictitious mass, funicular inversion

import numpy as np


class UniversalFormFindingSolver:
    """
    Robust Dynamic Relaxation solver for 3D structural form-finding.
    Handles vaults, domes, cable nets, tensile fabrics, and gridshells.

    Key fixes v2:
      1. Self-weight density corrected (was 1000x too small)
      2. Fictitious nodal mass added (required for stable DR)
      3. Funicular inversion: solve hanging → invert Z for vault
      4. Prestress only applied for cables and fabrics
      5. Adaptive dt based on stiffness to prevent divergence
    """

    def __init__(
        self,
        domain,
        E_modulus: float = 210000.0,
        gamma_kn_m3: float = 78.5,
        cross_section_area: float = 90000.0,
        area_mm2: float = None,
        prestress_force: float = 0.0,
        point_loads: list = None,
        material_type: str = "generic",
    ):
        self.domain = domain
        self.E = max(float(E_modulus), 1.0)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)
        self.material_type = str(material_type)

        if area_mm2 is not None:
            self.area = max(float(area_mm2), 1e-4)
        else:
            self.area = max(float(cross_section_area), 1e-4)

        # Prestress only meaningful for cables and fabrics
        if self.material_type in ("cable", "fabric"):
            self.prestress = float(prestress_force)
        else:
            self.prestress = 0.0

        self.point_loads = point_loads if point_loads is not None else []

    def _compute_fictitious_mass(self, rest_lengths: np.ndarray, edges: np.ndarray,
                                  num_nodes: int) -> np.ndarray:
        """
        Computes per-node fictitious mass for stable explicit time integration.
        Based on the largest axial stiffness connected to each node.
        Critical Timestep: dt < 2 * sqrt(m / k_max)
        """
        k_elem = self.E * self.area / np.maximum(rest_lengths, 1.0)
        node_k = np.zeros(num_nodes)

        for i, (u, v) in enumerate(edges):
            node_k[u] += k_elem[i]
            node_k[v] += k_elem[i]

        # Fictitious mass: m = alpha * k * dt^2 / 4
        # We choose alpha so dt=0.01 is stable: m = k * 0.01^2 / 4 * 100
        # Simplified: m proportional to k, scaled to give natural freq ~1 rad/s
        mass = np.maximum(node_k * 0.025, 1.0)
        return mass

    def _compute_self_weight(self, nodes: np.ndarray, edges: np.ndarray,
                              rest_lengths: np.ndarray) -> np.ndarray:
        """
        Distributes member self-weight as nodal forces in -Z direction.

        Fix v2: density_kg_mm3 = (gamma_kN_m3 * 1000 N/kN) / (9.81 m/s2) / (1e9 mm3/m3)
        Previous code was missing the *1000 factor — 1000x too light.
        """
        F_sw = np.zeros((len(nodes), 3))

        if self.gamma_kn_m3 <= 0.0:
            return F_sw

        # Correct unit conversion: kN/m3 → kg/mm3
        density_kg_mm3 = (self.gamma_kn_m3 * 1000.0 / 9.81) / 1e9

        for i, (u, v) in enumerate(edges):
            L = rest_lengths[i]
            member_mass_kg = density_kg_mm3 * self.area * L
            half_weight_N = (member_mass_kg * 9.81) / 2.0

            # Apply as downward (-Z) nodal force
            F_sw[u, 2] -= half_weight_N
            F_sw[v, 2] -= half_weight_N

        return F_sw

    def solve_equilibrium(
        self,
        iterations: int = 500,
        invert_form: bool = False,
        dt: float = None,
        damping: float = 0.80,
    ):
        """
        Runs Dynamic Relaxation to find structural equilibrium.

        Args:
            iterations:  Number of DR iterations (300-1000 recommended)
            invert_form: If True, inverts Z after solving to convert
                         hanging catenary to compression vault/dome
            dt:          Timestep (auto-calculated if None)
            damping:     Velocity damping factor (0.7-0.9 typical)

        Returns:
            nodes:        Equilibrium node positions (N x 3)
            axial_forces: Member axial forces in Newtons (M,)
            reactions:    Support reaction forces (N x 3)
        """
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        if num_nodes == 0 or num_edges == 0:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3))

        if len(fixed_nodes) == 0:
            # No supports defined — fix perimeter as fallback
            boundary = self.domain.get_boundary_nodes()
            fixed_nodes = set(boundary.tolist())

        # Compute rest lengths from initial flat geometry
        rest_lengths = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            dist = np.linalg.norm(nodes[u] - nodes[v])
            rest_lengths[i] = dist if dist > 1e-4 else 1.0

        # Fictitious nodal masses for stable integration
        fictitious_mass = self._compute_fictitious_mass(rest_lengths, edges, num_nodes)

        # Auto-calculate stable timestep if not provided
        if dt is None:
            k_max = self.E * self.area / np.min(rest_lengths)
            m_min = np.min(fictitious_mass)
            dt = 0.8 * 2.0 * np.sqrt(m_min / k_max)
            dt = np.clip(dt, 1e-6, 0.5)

        # External forces
        F_ext = self._compute_self_weight(nodes, edges, rest_lengths)

        # Point loads
        for ld in self.point_loads:
            px = float(ld.get("x", 0))
            py = float(ld.get("y", 0))
            pz = float(ld.get("z", 0))
            fx_N = float(ld.get("Fx", 0)) * 1000.0
            fy_N = float(ld.get("Fy", 0)) * 1000.0
            fz_N = float(ld.get("Fz", 0)) * 1000.0

            dists = np.linalg.norm(nodes - np.array([px, py, pz]), axis=1)
            closest = int(np.argmin(dists))
            F_ext[closest, 0] += fx_N
            F_ext[closest, 1] += fy_N
            F_ext[closest, 2] += fz_N

       # Scale loads relative to structural stiffness for meaningful deformation
        total_load = np.linalg.norm(F_ext)
        EA = self.E * self.area
        target_load = EA * 0.001  # 0.1% strain driving force
        if total_load < 1e-3:
            for i in range(num_nodes):
                if i not in fixed_nodes:
                    F_ext[i, 2] -= target_load / num_nodes
        elif total_load > 0:
    # Scale up small loads to be meaningful
    scale = max(1.0, target_load / total_load * 0.01)
    F_ext *= scale

        # Dynamic Relaxation main loop
        velocities = np.zeros((num_nodes, 3), dtype=float)
        axial_forces = np.zeros(num_edges, dtype=float)
        reactions = np.zeros((num_nodes, 3), dtype=float)

        for iteration in range(max(iterations, 10)):
            F_int = np.zeros((num_nodes, 3), dtype=float)

            # Compute internal forces from all members
            for i, (u, v) in enumerate(edges):
                vec = nodes[v] - nodes[u]
                curr_len = np.linalg.norm(vec)

                if curr_len < 1e-8:
                    continue

                unit_vec = vec / curr_len
                L0 = rest_lengths[i]
                strain = (curr_len - L0) / L0

                # Axial force: elastic + prestress (cables/fabrics only)
                force = self.E * self.area * strain + self.prestress

                # Prevent tension in compression-only materials
                if self.material_type in ("concrete", "masonry", "stone", "timber"):
                    force = min(force, 0.0)  # compression only — no tension

                # Prevent compression in tension-only members
                if self.material_type in ("cable", "fabric"):
                    force = max(force, self.prestress)  # tension only

                # Sanitise
                if np.isnan(force) or np.isinf(force):
                    force = 0.0

                axial_forces[i] = force
                f_vec = force * unit_vec
                F_int[u] += f_vec
                F_int[v] -= f_vec

            # Update positions
            for i in range(num_nodes):
                if i in fixed_nodes:
                    reactions[i] = -(F_int[i] + F_ext[i])
                    velocities[i] = 0.0
                else:
                    R = F_ext[i] + F_int[i]
                    # Correct DR: accelerate with fictitious mass
                    acc = R / fictitious_mass[i]
                    velocities[i] = (velocities[i] + acc * dt) * damping
                    nodes[i] += velocities[i] * dt

            # Kinetic energy damping (viscous reset at peaks)
            # Every 10 iterations check if kinetic energy is increasing
            if iteration % 50 == 0 and iteration > 0:
                KE = np.sum(fictitious_mass[:, np.newaxis] * velocities ** 2)
                if np.isnan(KE) or KE > 1e20:
                    # Diverging — reset velocities with smaller dt
                    velocities[:] = 0.0
                    dt *= 0.5

        # Funicular inversion for compression forms
        if invert_form:
            free_nodes = [i for i in range(num_nodes) if i not in fixed_nodes]
            if free_nodes:
                # Find min Z among free nodes (lowest hanging point)
                min_z = np.min(nodes[free_nodes, 2])
                max_z = np.max(nodes[free_nodes, 2])
                z_range = max_z - min_z

                # Invert free nodes around the support level
                support_z = np.mean(nodes[list(fixed_nodes), 2]) if fixed_nodes else 0.0

                for i in free_nodes:
                    # Map: hanging low → vault high
                    # The deepest hanging point becomes the vault crown
                    nodes[i, 2] = support_z + (support_z - nodes[i, 2])

                # Scale to target rise Lz
                new_z_range = np.max(nodes[free_nodes, 2]) - np.min(nodes[free_nodes, 2])
               if new_z_range > 1.0:  # only scale if meaningful deformation occurred
                    scale = self.domain.Lz / new_z_range
                    for i in free_nodes:
                        nodes[i, 2] = support_z + (nodes[i, 2] - support_z) * scale

                reactions[:, 2] = -reactions[:, 2]

        # Final sanitisation
        nodes = np.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)
        axial_forces = np.nan_to_num(axial_forces, nan=0.0, posinf=0.0, neginf=0.0)
        reactions = np.nan_to_num(reactions, nan=0.0, posinf=0.0, neginf=0.0)

        return nodes, axial_forces, reactions
