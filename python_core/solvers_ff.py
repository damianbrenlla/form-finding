# DBSW 3D Multi-Algorithm Form-Finding Engine
# Author: Damian Brenlla / DBSW 2026
# Pass 3 — Kinematic stability validation, prestress-assembled residual normalization,
#          edge-cable prestress application, and correct domain spacing ingestion.

import numpy as np


class ForceDensitySolver:
    def __init__(self, domain, E_modulus=210000.0, area_mm2=100.0, prestress_force=0.0, point_loads=None, gamma_kn_m3=78.5, **kwargs):
        self.domain = domain
        self.E = max(float(E_modulus), 1.0)
        self.area = max(float(area_mm2), 1e-4)
        self.prestress = float(prestress_force)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations=1, invert_form=False, **kwargs):
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

        P_ext = np.zeros((N_free, 3), dtype=float)

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
    def __init__(self, domain, E_modulus=33000.0, gamma_kn_m3=25.0, area_mm2=90000.0, point_loads=None, **kwargs):
        self.domain = domain
        self.E = max(float(E_modulus), 1.0)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)
        self.area = max(float(area_mm2), 1e-4)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations=1000, rel_tol=1e-4, **kwargs):
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        empty_diagnostics = {
            "method": "Vectorised Inverted Kinetic DR",
            "form_inverted": True,
            "iterations_run": 0,
            "converged": False,
            "relative_residual": 1.0,
        }

        if num_nodes == 0 or num_edges == 0:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3)), empty_diagnostics

        if len(fixed_nodes) < 3:
            raise ValueError("Kinematically unstable boundary conditions: Fewer than 3 fixed support nodes resolved.")

        fixed_pts = nodes[list(fixed_nodes)]
        vecs = fixed_pts[1:] - fixed_pts[0]
        if np.linalg.matrix_rank(vecs) < 2:
            raise ValueError("Kinematically unstable boundary conditions: Support nodes are collinear, leaving rotation unconstrained.")

        C = np.zeros((num_edges, num_nodes), dtype=float)
        for i, (u, v) in enumerate(edges):
            C[i, u] = -1.0
            C[i, v] = 1.0

        edge_vecs = nodes[edges[:, 1]] - nodes[edges[:, 0]]
        rest_lengths = np.linalg.norm(edge_vecs, axis=1)
        rest_lengths = np.where(rest_lengths < 1e-4, 1.0, rest_lengths)

        F_ext = np.zeros((num_nodes, 3), dtype=float)
        gamma_effective = self.gamma_kn_m3 if self.gamma_kn_m3 > 0 else 25.0
        density_kg_mm3 = (gamma_effective / 9.81) * 1e-6

        for i, (u, v) in enumerate(edges):
            L = rest_lengths[i]
            half_weight_N = ((density_kg_mm3 * self.area * L) * 9.81) / 2.0
            F_ext[u, 2] -= half_weight_N
            F_ext[v, 2] -= half_weight_N

        for ld in self.point_loads:
            px, py, pz = float(ld.get("x", 0)), float(ld.get("y", 0)), float(ld.get("z", 0))
            fz_N = float(ld.get("Fz", -10.0)) * 1000.0
            dists = np.linalg.norm(nodes - np.array([px, py, pz]), axis=1)
            closest_idx = int(np.argmin(dists))
            F_ext[closest_idx, 2] += fz_N

        total_ext_force_mag = np.max(np.linalg.norm(F_ext, axis=1)) if len(F_ext) > 0 else 1.0
        force_denom = max(total_ext_force_mag, 1.0)

        dt = 0.005
        k_edges = (self.E * self.area) / rest_lengths
        node_stiffness = np.abs(C.T) @ k_edges
        nodal_mass = np.maximum(0.5 * (dt ** 2) * node_stiffness * 2.5, 1e-6)[:, None]

        velocities = np.zeros((num_nodes, 3), dtype=float)
        prev_ke = 0.0
        free_mask = np.ones(num_nodes, dtype=bool)
        free_mask[list(fixed_nodes)] = False

        converged = False
        final_rel_res = 1.0
        iters_run = 0

        for it in range(max(iterations, 200)):
            iters_run = it + 1

            dXYZ = nodes[edges[:, 1]] - nodes[edges[:, 0]]
            curr_lengths = np.linalg.norm(dXYZ, axis=1)
            curr_lengths = np.where(curr_lengths < 1e-6, 1e-6, curr_lengths)
            unit_vecs = dXYZ / curr_lengths[:, None]

            strains = (curr_lengths - rest_lengths) / rest_lengths
            axial_forces = self.E * self.area * strains
            axial_forces = np.maximum(0.0, axial_forces)

            f_vecs = axial_forces[:, None] * unit_vecs
            F_int = np.zeros((num_nodes, 3), dtype=float)
            np.add.at(F_int, edges[:, 0], f_vecs)
            np.add.at(F_int, edges[:, 1], -f_vecs)

            R_residual = F_ext + F_int
            R_residual[~free_mask] = 0.0

            max_res = np.max(np.linalg.norm(R_residual, axis=1))
            final_rel_res = max_res / force_denom
            if final_rel_res < rel_tol and it > 50:
                converged = True
                break

            accel = R_residual / nodal_mass
            velocities[free_mask] += accel[free_mask] * dt

            ke = 0.5 * np.sum(nodal_mass[free_mask] * (velocities[free_mask] ** 2))

            if ke < prev_ke and it > 5:
                velocities[free_mask] = 0.0
                prev_ke = 0.0
            else:
                prev_ke = ke

            nodes[free_mask] += velocities[free_mask] * dt

        reactions = np.zeros((num_nodes, 3), dtype=float)
        reactions[~free_mask] = -(F_int[~free_mask] + F_ext[~free_mask])

        free_indices = np.where(free_mask)[0]
        support_z = np.mean(nodes[list(fixed_nodes), 2]) if fixed_nodes else 0.0

        nodes[free_indices, 2] = support_z + (support_z - nodes[free_indices, 2])
        reactions[:, 2] = -reactions[:, 2]
        axial_forces = -np.abs(axial_forces)

        diagnostics = {
            "method": "Vectorised Inverted Kinetic DR",
            "form_inverted": True,
            "iterations_run": iters_run,
            "converged": converged,
            "relative_residual": round(float(final_rel_res), 6),
        }

        return nodes, axial_forces, reactions, diagnostics


class UnderwoodDRSolver:
    def __init__(
        self, domain, E_modulus=210000.0, gamma_kn_m3=78.5, area_mm2=90000.0,
        prestress_force=0.0, prestress_warp_N_mm=0.0, prestress_weft_N_mm=0.0,
        edge_cable_prestress_N=0.0, point_loads=None, **kwargs
    ):
        self.domain = domain
        self.E = max(float(E_modulus), 1.0)
        self.gamma_kn_m3 = max(float(gamma_kn_m3), 0.0)
        self.area = max(float(area_mm2), 1e-4)
        self.prestress = float(prestress_force)
        self.prestress_warp_N_mm = float(prestress_warp_N_mm)
        self.prestress_weft_N_mm = float(prestress_weft_N_mm)
        self.edge_cable_prestress_N = float(edge_cable_prestress_N)
        self.point_loads = point_loads if point_loads is not None else []

    def solve_equilibrium(self, iterations=1000, rel_tol=1e-4, **kwargs):
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        empty_diagnostics = {
            "method": "Vectorised Underwood Kinetic DR",
            "form_inverted": False,
            "iterations_run": 0,
            "converged": False,
            "relative_residual": 1.0,
        }

        if num_nodes == 0 or num_edges == 0:
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3)), empty_diagnostics

        C = np.zeros((num_edges, num_nodes), dtype=float)
        for i, (u, v) in enumerate(edges):
            C[i, u] = -1.0
            C[i, v] = 1.0

        edge_vecs = nodes[edges[:, 1]] - nodes[edges[:, 0]]
        rest_lengths = np.linalg.norm(edge_vecs, axis=1)
        rest_lengths = np.where(rest_lengths < 1e-4, 1.0, rest_lengths)

        # CRITICAL FIX: Ingest actual domain grid spacing instead of hardcoded 100 mm
        dy_spacing = float(getattr(self.domain, "dy", self.domain.Ly / max(self.domain.ny, 1)))
        dx_spacing = float(getattr(self.domain, "dx", self.domain.Lx / max(self.domain.nx, 1)))

        prestress_array = np.full(num_edges, self.prestress, dtype=float)
        if self.prestress_warp_N_mm > 0.0 or self.prestress_weft_N_mm > 0.0:
            dx = np.abs(edge_vecs[:, 0])
            dy = np.abs(edge_vecs[:, 1])
            is_warp = dx >= dy
            prestress_array = np.where(
                is_warp,
                self.prestress_warp_N_mm * dy_spacing,
                self.prestress_weft_N_mm * dx_spacing
            )

        # CRITICAL FIX: Apply perimeter edge-cable prestress to topological boundary edges
        if self.edge_cable_prestress_N > 0.0:
            Lx, Ly = self.domain.Lx, self.domain.Ly
            tol = 1e-3
            for i, (u, v) in enumerate(edges):
                xu, yu = self.domain.nodes[u, 0], self.domain.nodes[u, 1]
                xv, yv = self.domain.nodes[v, 0], self.domain.nodes[v, 1]
                u_bound = (xu < tol or xu > Lx - tol or yu < tol or yu > Ly - tol)
                v_bound = (xv < tol or xv > Lx - tol or yv < tol or yv > Ly - tol)
                if u_bound and v_bound:
                    prestress_array[i] += self.edge_cable_prestress_N

        unit_vecs_0 = edge_vecs / rest_lengths[:, None]
        f_prestress_vecs = prestress_array[:, None] * unit_vecs_0
        F_prestress_nodal = np.zeros((num_nodes, 3), dtype=float)
        np.add.at(F_prestress_nodal, edges[:, 0], f_prestress_vecs)
        np.add.at(F_prestress_nodal, edges[:, 1], -f_prestress_vecs)

        F_ext = np.zeros((num_nodes, 3), dtype=float)

        if self.gamma_kn_m3 > 0.0:
            density_kg_mm3 = (self.gamma_kn_m3 / 9.81) * 1e-6
            for i, (u, v) in enumerate(edges):
                L = rest_lengths[i]
                half_weight_N = ((density_kg_mm3 * self.area * L) * 9.81) / 2.0
                F_ext[u, 2] -= half_weight_N
                F_ext[v, 2] -= half_weight_N

        for ld in self.point_loads:
            px, py, pz = float(ld.get("x", 0)), float(ld.get("y", 0)), float(ld.get("z", 0))
            fx_N = float(ld.get("Fx", 0)) * 1000.0
            fy_N = float(ld.get("Fy", 0)) * 1000.0
            fz_N = float(ld.get("Fz", 0)) * 1000.0

            dists = np.linalg.norm(nodes - np.array([px, py, pz]), axis=1)
            closest_idx = int(np.argmin(dists))
            F_ext[closest_idx] += [fx_N, fy_N, fz_N]

        total_ext_force_mag = np.max(np.linalg.norm(F_ext, axis=1)) if len(F_ext) > 0 else 1.0
        total_prestress_nodal_mag = np.max(np.linalg.norm(F_prestress_nodal, axis=1)) if len(F_prestress_nodal) > 0 else 1.0
        force_denom = max(total_ext_force_mag, total_prestress_nodal_mag, 1.0)

        dt = 0.005
        k_edges = (self.E * self.area) / rest_lengths
        node_stiffness = np.abs(C.T) @ k_edges
        nodal_mass = np.maximum(0.5 * (dt ** 2) * node_stiffness * 2.0, 1e-6)[:, None]

        velocities = np.zeros((num_nodes, 3), dtype=float)
        prev_ke = 0.0
        free_mask = np.ones(num_nodes, dtype=bool)
        free_mask[list(fixed_nodes)] = False

        converged = False
        final_rel_res = 1.0
        iters_run = 0

        for it in range(max(iterations, 200)):
            iters_run = it + 1

            dXYZ = nodes[edges[:, 1]] - nodes[edges[:, 0]]
            curr_lengths = np.linalg.norm(dXYZ, axis=1)
            curr_lengths = np.where(curr_lengths < 1e-6, 1e-6, curr_lengths)
            unit_vecs = dXYZ / curr_lengths[:, None]

            strains = (curr_lengths - rest_lengths) / rest_lengths
            axial_forces = (self.E * self.area * strains) + prestress_array

            if str(getattr(self.domain, "material_type", "")).lower() in ("membrane", "fabric"):
                axial_forces = np.maximum(0.0, axial_forces)

            f_vecs = axial_forces[:, None] * unit_vecs
            F_int = np.zeros((num_nodes, 3), dtype=float)
            np.add.at(F_int, edges[:, 0], f_vecs)
            np.add.at(F_int, edges[:, 1], -f_vecs)

            R_residual = F_ext + F_int
            R_residual[~free_mask] = 0.0

            max_res = np.max(np.linalg.norm(R_residual, axis=1))
            final_rel_res = max_res / force_denom
            if final_rel_res < rel_tol and it > 50:
                converged = True
                break

            accel = R_residual / nodal_mass
            velocities[free_mask] += accel[free_mask] * dt

            ke = 0.5 * np.sum(nodal_mass[free_mask] * (velocities[free_mask] ** 2))

            if ke < prev_ke and it > 5:
                velocities[free_mask] = 0.0
                prev_ke = 0.0
            else:
                prev_ke = ke

            nodes[free_mask] += velocities[free_mask] * dt

        reactions = np.zeros((num_nodes, 3), dtype=float)
        reactions[~free_mask] = -(F_int[~free_mask] + F_ext[~free_mask])

        diagnostics = {
            "method": "Vectorised Underwood Kinetic DR",
            "form_inverted": False,
            "iterations_run": iters_run,
            "converged": converged,
            "relative_residual": round(float(final_rel_res), 6),
        }

        return nodes, axial_forces, reactions, diagnostics


class FormFindingSolverFactory:
    @staticmethod
    def create(material_type, domain, mat_props, **kwargs):
        mat_type = str(material_type).lower()

        if mat_type in ("cables", "cable"):
            return ForceDensitySolver(domain=domain, E_modulus=mat_props.get("E", 160000.0), **kwargs)

        elif mat_type in ("concrete", "masonry", "stone"):
            return InvertedGravityDRSolver(domain=domain, E_modulus=mat_props.get("E", 33000.0), **kwargs)

        else:
            return UnderwoodDRSolver(domain=domain, E_modulus=mat_props.get("E", 210000.0), **kwargs)


class UniversalFormFindingSolver(UnderwoodDRSolver):
    pass
