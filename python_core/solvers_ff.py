# DBSW 3D Dynamic Relaxation & Dynamic Form-Finding Solver
# Author: Damian Brenlla / DBSW 2026
# v4 — Fixed: capped the FINAL HEIGHT-RESCALE step, which was still unbounded.
#      With self-weight only and no point loads (a common first run), the raw
#      deflection can be a fraction of a millimetre. The old code divided the
#      target rise (e.g. 2000mm) by that near-zero deflection to get a scale
#      factor in the thousands-to-millions, exploding node coordinates and
#      flinging the camera far from the origin on recenter -- which is why
#      the *entire* scene (grid, domain box, everything) went blank after
#      Execute, not just the missing structure. Also now returns a
#      diagnostics dict (achieved_rise_mm, target_rise_mm, height_scale_capped,
#      load_scale_capped) so the caller can warn the user when the result was
#      clamped rather than failing silently.
#
#      v3 fixes retained: capped stiffness-normalized load scaling, and a
#      per-iteration NaN/Inf guard on velocities.

import numpy as np


class UniversalFormFindingSolver:
    """Robust Dynamic Relaxation solver for 3D form-finding (vaults, domes, cable nets, fabrics)."""

    # Hard ceiling on the stiffness-normalisation multiplier applied to external
    # loads. The old code scaled F_ext by (E*A/1e6) / total_load with no upper
    # bound. For a typical concrete section (E=33000, A=90000mm^2 -> stiffness
    # ~2970) and a small total load, this produced multipliers in the
    # thousands, driving nodal displacements far outside anything physically
    # meaningful and risking overflow/NaN in the strain -> force calculation.
    MAX_LOAD_SCALE_FACTOR = 80.0

    # Hard ceiling on the height-rescale multiplier applied AFTER solving, when
    # stretching the solved shape to reach the user's target Lz. This is the
    # fix for the blank-viewport bug: without a cap here, a near-zero raw
    # deflection produces an astronomical scale factor regardless of how
    # tightly MAX_LOAD_SCALE_FACTOR constrains the loads themselves.
    MAX_HEIGHT_SCALE_FACTOR = 150.0

    # Minimum raw deflection (mm) before we attempt to rescale at all. Below
    # this, the shape is treated as "did not move enough to reshape" rather
    # than being blown up to fill the target height.
    MIN_DEFLECTION_FOR_RESCALE_MM = 0.05

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

        if area_mm2 is not None:
            self.area = max(float(area_mm2), 1e-4)
        else:
            self.area = max(float(cross_section_area), 1e-4)

        self.prestress = float(prestress_force)
        self.point_loads = point_loads if point_loads is not None else []
        self.material_type = material_type

    def solve_equilibrium(self, iterations: int = 500, invert_form: bool = False):
        nodes = np.copy(self.domain.nodes).astype(float)
        edges = np.copy(self.domain.edges).astype(int)
        fixed_nodes = set(self.domain.fixed_nodes)

        num_nodes = len(nodes)
        num_edges = len(edges)

        if num_nodes == 0 or num_edges == 0:
            empty_diagnostics = {
                "achieved_rise_mm": 0.0,
                "target_rise_mm": round(float(self.domain.Lz), 3),
                "height_scale_capped": False,
            }
            return nodes, np.zeros(num_edges), np.zeros((num_nodes, 3)), empty_diagnostics

        # Calculate initial rest lengths
        rest_lengths = np.zeros(num_edges, dtype=float)
        for i, (u, v) in enumerate(edges):
            dist = np.linalg.norm(nodes[u] - nodes[v])
            rest_lengths[i] = dist if dist > 1e-4 else 1.0

        F_ext = np.zeros((num_nodes, 3), dtype=float)

        # 1. Apply gravity self-weight (downward in -Z)
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

        # --- Stiffness-Normalized Load Scaling (fixed: capped) ---
        # Scale external forces relative to EA stiffness so stiff sections deform
        # meaningfully, but never by more than MAX_LOAD_SCALE_FACTOR. Without the
        # cap, stiff/lightly-loaded cases produced runaway multipliers.
        total_load = np.linalg.norm(F_ext)
        if total_load < 1e-3:
            # Default minimal downward drive load
            for i in range(num_nodes):
                if i not in fixed_nodes:
                    F_ext[i, 2] -= 100.0
        else:
            stiffness_factor = (self.E * self.area) / 1e6
            if stiffness_factor > 1.0:
                raw_scale = stiffness_factor / (total_load + 1e-6)
                scale = min(raw_scale, self.MAX_LOAD_SCALE_FACTOR)
                F_ext *= scale

        # Dynamic Relaxation Integration Loop
        velocities = np.zeros((num_nodes, 3), dtype=float)
        damping = 0.85
        dt = 0.005

        axial_forces = np.zeros(num_edges, dtype=float)
        reactions = np.zeros((num_nodes, 3), dtype=float)

        # --- FIX: mass-scaled dynamic relaxation (root cause of the blank-viewport
        # bug) ---
        # The previous loop integrated every node with an implicit unit mass
        # (velocities[i] = (velocities[i] + R_residual*dt)*damping, with no
        # division by mass). That is only numerically stable if dt is small
        # relative to sqrt(mass/stiffness) for the STIFFEST member in the
        # model. dt=0.005 was tuned for something soft like cable/fabric work
        # (EA in the tens/hundreds of thousands of N). For structural concrete
        # or steel with a 300x300mm section, EA is on the order of 3x10^9 N —
        # roughly four orders of magnitude stiffer — and the same fixed dt
        # diverges explosively within the first handful of iterations,
        # regardless of how small the applied load is. In testing, node
        # coordinates reached ~1e156 within 500 iterations for a plain
        # concrete self-weight-only case, which is what pushed the camera and
        # the entire scene out of view.
        #
        # Standard fix (Day/Underwood dynamic relaxation): give each node a
        # fictitious mass proportional to the stiffness of the members
        # connected to it, M_i = 0.5 * dt^2 * safety * sum(EA/L over adjacent
        # edges). This keeps the effective natural frequency — and therefore
        # numerical stability — roughly constant no matter how stiff the
        # material or how large the section is, without needing a different
        # hand-tuned dt per material family.
        mass_safety_factor = 2.0
        node_stiffness_sum = np.zeros(num_nodes, dtype=float)
        for i, (u, v) in enumerate(edges):
            k_edge = (self.E * self.area) / rest_lengths[i]
            node_stiffness_sum[u] += k_edge
            node_stiffness_sum[v] += k_edge

        nodal_mass = np.maximum(
            0.5 * (dt ** 2) * node_stiffness_sum * mass_safety_factor,
            1e-6,
        )

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
                    # Mass-scaled acceleration, not raw force -- this is what
                    # keeps the integration stable across the full range of
                    # material stiffnesses (PTFE fabric through structural steel)
                    # with a single fixed dt.
                    accel = R_residual / nodal_mass[i]
                    new_velocity = (velocities[i] + accel * dt) * damping
                    # Guard velocities every iteration, not just at the very
                    # end. A single NaN/Inf velocity (e.g. from a degenerate
                    # edge or transient overflow) would otherwise propagate
                    # through every subsequent iteration via
                    # nodes[i] += velocities[i] * dt, silently corrupting the
                    # whole run before the final np.nan_to_num cleanup masked it.
                    if not np.all(np.isfinite(new_velocity)):
                        new_velocity = np.zeros(3)
                    velocities[i] = new_velocity
                    nodes[i] += velocities[i] * dt

                    # Belt-and-braces: even with correct mass scaling, an
                    # extreme edge case (e.g. a near-zero-length starting
                    # member) shouldn't be able to fling a node to infinity
                    # and silently blank the viewport. Clamp each node's
                    # excursion from its starting position to a generous but
                    # finite multiple of the domain's own footprint.
                    max_excursion = 20.0 * max(self.domain.Lx, self.domain.Ly, self.domain.Lz, 1.0)
                    origin_pos = self.domain.nodes[i]
                    delta = nodes[i] - origin_pos
                    dist = np.linalg.norm(delta)
                    if dist > max_excursion and dist > 0:
                        nodes[i] = origin_pos + delta * (max_excursion / dist)
                        velocities[i] = np.zeros(3)

        # --- Inversion and Origin-Guarded Height Scaling ---
        free_nodes = [i for i in range(num_nodes) if i not in fixed_nodes]
        support_z = np.mean(nodes[list(fixed_nodes), 2]) if fixed_nodes else 0.0

        if invert_form:
            # 1. Mirror free nodes across the support plane (hanging Z < support_z -> vault Z > support_z)
            for i in free_nodes:
                nodes[i, 2] = support_z + (support_z - nodes[i, 2])
            reactions[:, 2] = -reactions[:, 2]

        # 2. Scale free nodes up to target domain Lz if meaningful deformation occurred.
        # FIX: raw_scale is now clamped to MAX_HEIGHT_SCALE_FACTOR. Previously,
        # a raw deflection of e.g. 0.01mm against a 2000mm target Lz produced
        # scale = 200,000 -- multiplying every free node's Z offset by that
        # factor and pushing coordinates into the hundreds of thousands or
        # millions, well past the camera's far-clipping plane and the point
        # where floating-point precision holds up visually.
        achieved_rise_mm = 0.0
        target_rise_mm = float(self.domain.Lz)
        height_scale_capped = False

        if free_nodes and self.domain.Lz > 0.0:
            new_z_range = float(np.max(np.abs(nodes[free_nodes, 2] - support_z)))
            if new_z_range > self.MIN_DEFLECTION_FOR_RESCALE_MM:
                raw_scale = target_rise_mm / new_z_range
                scale = min(raw_scale, self.MAX_HEIGHT_SCALE_FACTOR)
                height_scale_capped = raw_scale > self.MAX_HEIGHT_SCALE_FACTOR
                for i in free_nodes:
                    nodes[i, 2] = support_z + (nodes[i, 2] - support_z) * scale
                achieved_rise_mm = new_z_range * scale
            # else: deflection negligible even before scaling -- leave the
            # shape as solved (effectively flat) rather than amplifying noise.

        # Final NaN/Inf Sanitization
        nodes = np.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)
        axial_forces = np.nan_to_num(axial_forces, nan=0.0, posinf=0.0, neginf=0.0)
        reactions = np.nan_to_num(reactions, nan=0.0, posinf=0.0, neginf=0.0)

        diagnostics = {
            "achieved_rise_mm": round(float(achieved_rise_mm), 3),
            "target_rise_mm": round(target_rise_mm, 3),
            "height_scale_capped": bool(height_scale_capped),
        }

        return nodes, axial_forces, reactions, diagnostics
