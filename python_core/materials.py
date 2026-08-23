# DBSW Eurocode & Form-Finding Multi-Material Registry
# Author: Damian Brenlla / DBSW 2026


class FormFindingMaterialRegistry:
    """Centralized Registry for Structural Materials adhering to Eurocodes (EC2, EC3, EC5, EC6)
    plus Tensile Cables, Architectural Fabrics, and Custom Isotropic Grade logic.
    """

    STEEL_GRADES = {
        "S235": {"E": 210000.0, "nu": 0.30, "f_k": 235.0, "gamma_M": 1.00, "gamma_kn_m3": 78.5},
        "S275": {"E": 210000.0, "nu": 0.30, "f_k": 275.0, "gamma_M": 1.00, "gamma_kn_m3": 78.5},
        "S355": {"E": 210000.0, "nu": 0.30, "f_k": 355.0, "gamma_M": 1.00, "gamma_kn_m3": 78.5},
        "S460": {"E": 210000.0, "nu": 0.30, "f_k": 460.0, "gamma_M": 1.00, "gamma_kn_m3": 78.5},
    }

    CONCRETE_GRADES = {
        "C20/25": {"E": 30000.0, "nu": 0.20, "f_k": 20.0, "gamma_M": 1.50, "alpha_cc": 0.85, "gamma_kn_m3": 25.0},
        "C25/30": {"E": 31500.0, "nu": 0.20, "f_k": 25.0, "gamma_M": 1.50, "alpha_cc": 0.85, "gamma_kn_m3": 25.0},
        "C30/37": {"E": 33000.0, "nu": 0.20, "f_k": 30.0, "gamma_M": 1.50, "alpha_cc": 0.85, "gamma_kn_m3": 25.0},
        "C35/45": {"E": 34000.0, "nu": 0.20, "f_k": 35.0, "gamma_M": 1.50, "alpha_cc": 0.85, "gamma_kn_m3": 25.0},
        "C40/50": {"E": 35000.0, "nu": 0.20, "f_k": 40.0, "gamma_M": 1.50, "alpha_cc": 0.85, "gamma_kn_m3": 25.0},
        "C50/60": {"E": 37000.0, "nu": 0.20, "f_k": 50.0, "gamma_M": 1.50, "alpha_cc": 0.85, "gamma_kn_m3": 25.0},
    }

    TIMBER_GRADES = {
        "C16":   {"E": 8000.0,  "nu": 0.35, "f_k": 16.0, "gamma_M": 1.30, "gamma_kn_m3": 3.7},
        "C24":   {"E": 11000.0, "nu": 0.35, "f_k": 24.0, "gamma_M": 1.30, "gamma_kn_m3": 4.2},
        "C30":   {"E": 12000.0, "nu": 0.35, "f_k": 30.0, "gamma_M": 1.30, "gamma_kn_m3": 4.6},
        "GL24h": {"E": 11500.0, "nu": 0.35, "f_k": 24.0, "gamma_M": 1.25, "gamma_kn_m3": 3.9},
        "GL28h": {"E": 12600.0, "nu": 0.35, "f_k": 28.0, "gamma_M": 1.25, "gamma_kn_m3": 4.2},
        "GL32h": {"E": 13700.0, "nu": 0.35, "f_k": 32.0, "gamma_M": 1.25, "gamma_kn_m3": 4.4},
    }

    CABLE_GRADES = {
        "Steel Cable (7x19)":  {"E": 110000.0, "nu": 0.30, "f_k": 1570.0, "gamma_M": 1.15, "gamma_kn_m3": 78.5},
        "High-Tensile Wire":  {"E": 160000.0, "nu": 0.30, "f_k": 1860.0, "gamma_M": 1.15, "gamma_kn_m3": 78.5},
        "Stainless Strand":   {"E": 130000.0, "nu": 0.30, "f_k": 1450.0, "gamma_M": 1.15, "gamma_kn_m3": 79.0},
    }

    FABRIC_GRADES = {
        "PTFE Architectural": {"E": 1500.0, "nu": 0.25, "f_k": 80.0, "gamma_M": 1.40, "gamma_kn_m3": 14.0},
        "PVC-Coated Polyester": {"E": 900.0,  "nu": 0.25, "f_k": 50.0, "gamma_M": 1.40, "gamma_kn_m3": 11.0},
        "ETFE Membrane":        {"E": 650.0,  "nu": 0.40, "f_k": 40.0, "gamma_M": 1.30, "gamma_kn_m3": 17.5},
    }

    MASONRY_GAMMA_KN_M3 = 19.0
    STONE_GAMMA_KN_M3 = 26.0

    @classmethod
    def calculate_masonry_fk(cls, fb: float, fm: float, K: float = 0.55) -> float:
        """BS EN 1996-1-1 (EC6) Characteristic Compressive Strength: f_k = K * fb^0.7 * fm^0.3."""
        return K * (fb ** 0.7) * (fm ** 0.3)

    @classmethod
    def resolve_properties(cls, payload: dict) -> dict:
        mat_type = payload.get("material_type", "steel")

        if mat_type == "steel":
            grade = payload.get("material_grade", "S355")
            props = cls.STEEL_GRADES.get(grade, cls.STEEL_GRADES["S355"])
            return {
                "material_type": "steel",
                "material_name": f"Steel {grade}",
                "E": props["E"],
                "nu": props["nu"],
                "f_k": props["f_k"],
                "f_d": props["f_k"] / props["gamma_M"],
                "gamma_kn_m3": props["gamma_kn_m3"],
            }

        elif mat_type == "concrete":
            grade = payload.get("material_grade", "C30/37")
            props = cls.CONCRETE_GRADES.get(grade, cls.CONCRETE_GRADES["C30/37"])
            f_cd = props["alpha_cc"] * props["f_k"] / props["gamma_M"]
            return {
                "material_type": "concrete",
                "material_name": f"Concrete {grade}",
                "E": props["E"],
                "nu": props["nu"],
                "f_k": props["f_k"],
                "f_d": f_cd,
                "gamma_kn_m3": props["gamma_kn_m3"],
            }

        elif mat_type == "timber":
            grade = payload.get("material_grade", "C24")
            props = cls.TIMBER_GRADES.get(grade, cls.TIMBER_GRADES["C24"])
            return {
                "material_type": "timber",
                "material_name": f"Timber {grade}",
                "E": props["E"],
                "nu": props["nu"],
                "f_k": props["f_k"],
                "f_d": props["f_k"] / props["gamma_M"],
                "gamma_kn_m3": props["gamma_kn_m3"],
            }

        elif mat_type == "cable":
            grade = payload.get("material_grade", "Steel Cable (7x19)")
            props = cls.CABLE_GRADES.get(grade, cls.CABLE_GRADES["Steel Cable (7x19)"])
            return {
                "material_type": "cable",
                "material_name": f"Cable {grade}",
                "E": props["E"],
                "nu": props["nu"],
                "f_k": props["f_k"],
                "f_d": props["f_k"] / props["gamma_M"],
                "gamma_kn_m3": props["gamma_kn_m3"],
            }

        elif mat_type == "fabric":
            grade = payload.get("material_grade", "PTFE Architectural")
            props = cls.FABRIC_GRADES.get(grade, cls.FABRIC_GRADES["PTFE Architectural"])
            return {
                "material_type": "fabric",
                "material_name": f"Fabric {grade}",
                "E": props["E"],
                "nu": props["nu"],
                "f_k": props["f_k"],
                "f_d": props["f_k"] / props["gamma_M"],
                "gamma_kn_m3": props["gamma_kn_m3"],
            }

        elif mat_type == "masonry":
            fb_val = float(payload.get("masonry_unit_fb", 20.0))
            fm_val = float(payload.get("mortar_fm", 6.0))
            fk_eff = cls.calculate_masonry_fk(fb_val, fm_val, K=0.55)
            E_eff = 1000.0 * fk_eff
            return {
                "material_type": "masonry",
                "material_name": f"Masonry (fb={fb_val}MPa, fm={fm_val}MPa)",
                "E": E_eff,
                "nu": 0.20,
                "f_k": fk_eff,
                "f_d": fk_eff / 1.5,
                "gamma_kn_m3": cls.MASONRY_GAMMA_KN_M3,
            }

        elif mat_type == "stone":
            fb_val = float(payload.get("masonry_unit_fb", 100.0))
            fm_val = float(payload.get("mortar_fm", 6.0))
            fk_eff = cls.calculate_masonry_fk(fb_val, fm_val, K=0.45)
            E_explicit = float(payload.get("stone_E", 35000.0))
            return {
                "material_type": "stone",
                "material_name": f"Natural Stone Masonry (fb={fb_val}MPa)",
                "E": E_explicit,
                "nu": 0.22,
                "f_k": fk_eff,
                "f_d": fk_eff / 1.5,
                "gamma_kn_m3": cls.STONE_GAMMA_KN_M3,
            }

        elif mat_type == "generic":
            E_val = float(payload.get("custom_E", 210000.0))
            nu_val = float(payload.get("custom_nu", 0.30))
            fk_val = float(payload.get("custom_fk", 355.0))
            gamma_val = float(payload.get("custom_gamma_kn_m3", 0.0))
            return {
                "material_type": "generic",
                "material_name": "Custom/Generic Material",
                "E": E_val,
                "nu": nu_val,
                "f_k": fk_val,
                "f_d": fk_val,
                "gamma_kn_m3": gamma_val,
            }

        raise ValueError(f"Unsupported material type: {mat_type}")
