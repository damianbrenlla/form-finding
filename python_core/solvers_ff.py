# DBSW 3D Multi-Algorithm Form-Finding Engine
# Author: Damian Brenlla / DBSW 2026
# Phase 1: Force Density Method (Cables), Compression-Constrained Inverted DR (Vaults/Masonry), Underwood DR (Generic)

import numpy as np


class ForceDensitySolver:
    """
    Direct Linear Solver using the Force Density Method (FDM) for Cable Nets.
    
    NOTE ON CATENARY DEAD LOAD:
    FDM is strictly linear for fixed force density ratios q = S / L. Element self-weight
    is incorporated deterministically as a lumped vertical RHS load vector f_sw.
    This solves exact spatial catenary equilibrium under dead load in a single linear pass
    without requiring iterative q-optimisation loops.
    """

    def __init__(self, domain, E_modulus: float = 210000.0, area_mm2: float = 100.0, prestress_force: float = 0.0, point_loads: list = None, gamma_kn_m3: float = 78.5, **kwargs):
        self.domain = domain
        self.E = max(float(E_modulus), 1.0)
        self.area = max(float(area_mm2), 1e-4)
        self.prestress = float(prestress_force)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations: int = 1, invert_form: bool = False):
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        empty_diagnostics = {
            "method": "Force Density Method (FDM)",
            "slack_cables": 0,
            "system_solved_linearly": False,
            "achieved_rise_mm": 0.0,
            "target_rise_mm": round(float(getattr(self.domain, "Lz", 0.0)), 3),
            "height_scale_capped": False,
        }

        if num_nodes == 0 or num_edges == 0:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3)), empty_diagnostics

        free_indices = [i for i in range(num_nodes) if i not in fixed_nodes]
        fixed_indices = sorted(list(fixed_nodes))

        if not free_indices:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3)), empty_diagnostics

        rest_lengths = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            dist = np.linalg.norm(nodes[u] - nodes[v])
            rest_lengths[i] = dist if dist > 1e-4 else 1.0

        base_q = max(self.prestress / np.mean(rest_lengths), 10.0) if self.prestress > 0 else 10.0
        q = np.full(num_edges, base_q, dtype=float)

        free_map = {node_idx: pos for pos, node_idx in enumerate(free_indices)}
        fixed_map = {node_idx: pos for pos, node_idx in enumerate(fixed_indices)}

        N_free = len(free_indices)
        N_fixed = len(fixed_indices)

        D = np.zeros((N_free, N_free), dtype=float)
        Df = np.zeros((N_free, N_fixed), dtype=float)
        free_nodes_set = set(free_indices)

        for k, (u, v) in enumerate(edges):
            q_k = q[k]

            if u in free_nodes_set:
                iu = free_map[u]
                D[iu, iu] += q_k
                if v in free_nodes_set:
                    iv = free_map[v]
                    D[iu, iv] -= q_k
                else:
                    iv_f = fixed_map[v]
                    Df[iu, iv_f] -= q_k

            if v in free_nodes_set:
                iv = free_map[v]
                D[iv, iv] += q_k
                if u in free_nodes_set:
                    iu = free_map[u]
                    D[iv, iu] -= q_k
                else:
                    iu_f = fixed_map[u]
                    Df[iv, iu_f] -= q_k

        # --- External and Self-Weight Load Assembly ---
        P_ext = np.zeros((N_free, 3), dtype=float)

        # Cable Dead Load (kN/m3 -> kg/mm3: gamma_kn_m3 * 1000 / 9.81 / 1e9 = gamma / 9.81 * 1e-6)
        if self.gamma_kn_m3 > 0.0:
            density_kg_mm3 = (self.gamma_kn_m3 / 9.81) * 1e-6
            for i, (u, v) in enumerate(edges):
                L = rest_lengths[i]
                member_mass_kg = density_kg_mm3 * self.area * L
                half_weight_N = (member_mass_kg * 9.81) / 2.0
                if u in free_map:
                    P_ext[free_map[u], 2] -= half_weight_N
                if v in free_map:
                    P_ext[free_map[v], 2] -= half_weight_N

        # External Point Loads
        for ld in self.point_loads:
            px, py, pz = float(ld.get("x", 0)), float(ld.get("y", 0)), float(ld.get("z", 0))
            fx_N = float(ld.get("Fx", 0)) * 1000.0
            fy_N = float(ld.get("Fy", 0)) * 1000.0
            fz_N = float(ld.get("Fz", 0)) * 1000.0

            dists = np.linalg.norm(nodes - np.array([px, py, pz]), axis=1)
            closest_idx = int(np.argmin(dists))
            if closest_idx in free_map:
                f_pos = free_map[closest_idx]
                P_ext[f_pos] += [fx_N, fy_N, fz_N]

        X_fixed = nodes[fixed_indices]
        RHS = P_ext - np.dot(Df, X_fixed)

        # Fast Pivot Conditioning Check (Prevents silent inversion of unconstrained systems)
        diag_D = np.abs(np.diag(D))
        if np.any(diag_D < 1e-12):
            raise ValueError("Singular force density matrix: cable network is under-constrained or missing boundary supports.")

        try:
            X_free = np.linalg.solve(D, RHS)
            nodes[free_indices] = X_free
        except np.linalg.LinAlgError:
            raise ValueError("Unstable boundary conditions: Force density matrix linear solve failed.")

        axial_forces = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            curr_len = np.linalg.norm(nodes[u] - nodes[v])
            axial_forces[i] = q[i] * curr_len

        reactions = np.zeros((num_nodes, 3), dtype=float)
        for i, (u, v) in enumerate(edges):
            vec = nodes[v] - nodes[u]
            curr_len = np.linalg.norm(vec)
            if curr_len > 1e-6:
                unit_vec = vec / curr_len
                f_vec = axial_forces[i] * unit_vec
                if u in fixed_nodes:
                    reactions[u] -= f_vec
                if v in fixed_nodes:
                    reactions[v] += f_vec

        slack_cables_count = int(np.sum(axial_forces <= 0.0))

        diagnostics = {
            "method": "Force Density Method (FDM)",
            "slack_cables": slack_cables_count,
            "system_solved_linearly": True,
            "achieved_rise_mm": 0.0,
            "target_rise_mm": round(float(getattr(self.domain, "Lz", 0.0)), 3),
            "height_scale_capped": False,
        }

        return nodes, axial_forces, reactions, diagnostics


class InvertedGravityDRSolver:
    """Compression-Constrained Inverted Dynamic Relaxation for Vaults, Domes & Masonry."""

    def __init__(self, domain, E_modulus: float = 33000.0, gamma_kn_m3: float = 25.0, area_mm2: float = 90000.0, point_loads: list = None, **kwargs):
        self.domain = domain
        self.E = max(float(E_modulus), 1.0)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)
        self.area = max(float(area_mm2), 1e-4)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations: int = 500, invert_form: bool = True):
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        empty_diagnostics = {
            "method": "Compression-Constrained Inverted DR",
            "form_inverted": True,
            "achieved_rise_mm": 0.0,
            "target_rise_mm": round(float(getattr(self.domain, "Lz", 0.0)), 3),
            "height_scale_capped": False,
        }

        if num_nodes == 0 or num_edges == 0:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3)), empty_diagnostics

        rest_lengths = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            dist = np.linalg.norm(nodes[u] - nodes[v])
            rest_lengths[i] = dist if dist > 1e-4 else 1.0

        F_ext = np.zeros((num_nodes, 3), dtype=float)

        # Correct unit conversion: kN/m3 -> kg/mm3 = gamma * 1000 / 9.81 / 1e9 = gamma / 9.81 * 1e-6
        gamma_effective = self.gamma_kn_m3 if self.gamma_kn_m3 > 0 else 25.0
        density_kg_mm3 = (gamma_effective / 9.81) * 1e-6
        for i, (u, v) in enumerate(edges):
            L = rest_lengths[i]
            member_mass_kg = density_kg_mm3 * self.area * L
            half_weight_N = (member_mass_kg * 9.81) / 2.0
            F_ext[u, 2] -= half_weight_N
            F_ext[v, 2] -= half_weight_N

        for ld in self.point_loads:
            px, py, pz = float(ld.get("x", 0)), float(ld.get("y", 0)), float(ld.get("z", 0))
            fz_N = float(ld.get("Fz", -10.0)) * 1000.0
            dists = np.linalg.norm(nodes - np.array([px, py, pz]), axis=1)
            closest_idx = int(np.argmin(dists))
            F_ext[closest_idx, 2] += fz_N

        velocities = np.zeros((num_nodes, 3), dtype=float)
        damping = 0.82
        dt = 0.005

        axial_forces = np.zeros(num_edges, dtype=float)
        reactions = np.zeros((num_nodes, 3), dtype=float)

        node_stiffness_sum = np.zeros(num_nodes, dtype=float)
        for i, (u, v) in enumerate(edges):
            k_edge = (self.E * self.area) / rest_lengths[i]
            node_stiffness_sum[u] += k_edge
            node_stiffness_sum[v] += k_edge

        nodal_mass = np.maximum(0.5 * (dt ** 2) * node_stiffness_sum * 2.5, 1e-6)

        for _ in range(max(iterations, 100)):
            F_int = np.zeros((num_nodes, 3), dtype=float)

            for i, (u, v) in enumerate(edges):
                vec = nodes[v] - nodes[u]
                curr_len = np.linalg.norm(vec)
                if curr_len < 1e-6:
                    continue
                unit_vec = vec / curr_len

                L0 = rest_lengths[i]
                strain = (curr_len - L0) / L0
                force = self.E * self.area * strain

                if force < 0.0:
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
                    accel = R_residual / nodal_mass[i]
                    velocities[i] = (velocities[i] + accel * dt) * damping
                    nodes[i] += velocities[i] * dt

        free_nodes = [i for i in range(num_nodes) if i not in fixed_nodes]
        support_z = np.mean(nodes[list(fixed_nodes), 2]) if fixed_nodes else 0.0

        for i in free_nodes:
            nodes[i, 2] = support_z + (support_z - nodes[i, 2])
        reactions[:, 2] = -reactions[:, 2]
        axial_forces = -np.abs(axial_forces)

        diagnostics = {
            "method": "Compression-Constrained Inverted DR",
            "form_inverted": True,
            "achieved_rise_mm": 0.0,
            "target_rise_mm": round(float(getattr(self.domain, "Lz", 0.0)), 3),
            "height_scale_capped": False,
        }

        return nodes, axial_forces, reactions, diagnostics


class UnderwoodDRSolver:
    """General Mass-Scaled Dynamic Relaxation Solver for Generic Isotropic Materials."""

    def __init__(self, domain, E_modulus: float = 210000.0, gamma_kn_m3: float = 78.5, area_mm2: float = 90000.0, prestress_force: float = 0.0, point_loads: list = None, **kwargs):
        self.domain = domain
        self.E = max(float(E_modulus), 1.0)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)
        self.area = max(float(area_mm2), 1e-4)
        self.prestress = float(prestress_force)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations: int = 500, invert_form: bool = False):
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        empty_diagnostics = {
            "method": "Underwood Dynamic Relaxation (DR)",
            "form_inverted": False,
            "achieved_rise_mm": 0.0,
            "target_rise_mm": round(float(getattr(self.domain, "Lz", 0.0)), 3),
            "height_scale_capped": False,
        }

        if num_nodes == 0 or num_edges == 0:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3)), empty_diagnostics

        rest_lengths = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            dist = np.linalg.norm(nodes[u] - nodes[v])
            rest_lengths[i] = dist if dist > 1e-4 else 1.0

        F_ext = np.zeros((num_nodes, 3), dtype=float)

        # Correct unit conversion: kN/m3 -> kg/mm3 = gamma * 1000 / 9.81 / 1e9 = gamma / 9.81 * 1e-6
        if self.gamma_kn_m3 > 0.0:
            density_kg_mm3 = (self.gamma_kn_m3 / 9.81) * 1e-6
            for i, (u, v) in enumerate(edges):
                L = rest_lengths[i]
                member_mass_kg = density_kg_mm3 * self.area * L
                half_weight_N = (member_mass_kg * 9.81) / 2.0
                F_ext[u, 2] -= half_weight_N
                F_ext[v, 2] -= half_weight_N

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

        velocities = np.zeros((num_nodes, 3), dtype=float)
        damping = 0.85
        dt = 0.005

        axial_forces = np.zeros(num_edges, dtype=float)
        reactions = np.zeros((num_nodes, 3), dtype=float)

        node_stiffness_sum = np.zeros(num_nodes, dtype=float)
        for i, (u, v) in enumerate(edges):
            k_edge = (self.E * self.area) / rest_lengths[i]
            node_stiffness_sum[u] += k_edge
            node_stiffness_sum[v] += k_edge

        nodal_mass = np.maximum(0.5 * (dt ** 2) * node_stiffness_sum * 2.0, 1e-6)

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
                    accel = R_residual / nodal_mass[i]
                    velocities[i] = (velocities[i] + accel * dt) * damping
                    nodes[i] += velocities[i] * dt

        diagnostics = {
            "method": "Underwood Dynamic Relaxation (DR)",
            "form_inverted": False,
            "achieved_rise_mm": 0.0,
            "target_rise_mm": round(float(getattr(self.domain, "Lz", 0.0)), 3),
            "height_scale_capped": False,
        }

        return nodes, axial_forces, reactions, diagnostics


class FormFindingSolverFactory:
    """Factory creating appropriate structural form-finding solver algorithms."""

    @staticmethod
    def create(material_type: str, domain, mat_props: dict, **kwargs):
        mat_type = str(material_type).lower()

        if mat_type in ("cables", "cable"):
            return ForceDensitySolver(domain=domain, E_modulus=mat_props.get("E", 160000.0), **kwargs)

        elif mat_type in ("concrete", "masonry", "stone"):
            return InvertedGravityDRSolver(domain=domain, E_modulus=mat_props.get("E", 33000.0), **kwargs)

        else:
            return UnderwoodDRSolver(domain=domain, E_modulus=mat_props.get("E", 210000.0), **kwargs)


class UniversalFormFindingSolver(UnderwoodDRSolver):
    """Backward-compatible alias pointing to UnderwoodDRSolver."""
    pass
