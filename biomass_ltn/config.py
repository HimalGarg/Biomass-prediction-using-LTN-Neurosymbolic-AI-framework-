"""Shared constants for the biomass LTN project."""

TARGET_NAMES = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]

TARGET_INDEX = {name: idx for idx, name in enumerate(TARGET_NAMES)}

PRIMITIVE_TARGETS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
]

DERIVED_RULES = {
    "Dry_Total_g": "Dry_Clover_g + Dry_Dead_g + Dry_Green_g",
    "GDM_g": "Dry_Clover_g + Dry_Green_g",
}

