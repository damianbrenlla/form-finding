# DBSW Universal Form-Finding Material Registry
# Author: Damian Brenlla / DBSW 2026

class FormFindingMaterialRegistry:
    """Registry supporting Solids, Cables, Membranes, and Fabrics."""

    MATERIALS = {
        # --- Structural Solids ---
        "S355": {"E": 210000.0, "nu": 0.30, "gamma_kn_m3": 78.5, "type": "isotropic"},
        "C30/37": {"E": 33000.0, "nu": 0.20, "gamma_kn_m3": 25.0, "type": "isotropic"},
        "C16_Timber": {"E": 8000.0, "nu": 0.35, "gamma_kn_m3": 3.7, "type": "isotropic"},
        "Masonry_EC6": {"E": 6000.0, "nu": 0.20, "gamma_kn_m3": 19.0, "type": "no_tension"},

        # --- Cables, Fabrics & Tension Membranes ---
        "Steel_Cable_7x19": {"E": 160000.0, "nu": 0.30, "gamma_kn_m3": 78.5, "type": "cable_only"},
        "PTFE_Architectural_Fabric": {"E": 1200.0, "nu": 0.40, "gamma_kn_m3": 15.0, "type": "membrane"},
        "ETFE_Foil": {"E": 220.0, "nu": 0.45, "gamma_kn_m3": 17.5, "type": "membrane"},
        "PVC_Coated_Polyester": {"E": 800.0, "nu": 0.35, "gamma_kn_m3": 11.0, "type": "membrane"},
        "Nylon_High_Tenacity": {"E": 3500.0, "nu": 0.39, "gamma_kn_m3": 11.2, "type": "cable_only"},
    }

    @classmethod
    def resolve_properties(cls, payload: dict) -> dict:
        mat_key = payload.get("material_grade", "S355")
        props = cls.MATERIALS.get(mat_key, cls.MATERIALS["S355"])
        return {
            "material_name": mat_key,
            "E": props["E"],
            "nu": props["nu"],
            "gamma_kn_m3": props["gamma_kn_m3"],
            "type": props["type"],
        }