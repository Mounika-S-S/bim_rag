from .extract_basic import extract_basic_info
from .extract_psets import extract_psets

def extract_wall_data(wall):
    basic = extract_basic_info(wall)
    psets = extract_psets(wall)

    fire_rating = None
    if "Pset_WallCommon" in psets:
        fire_rating = psets["Pset_WallCommon"].get("FireRating")

    return {
        **basic,
        "fire_rating": fire_rating,
        "psets": psets
    }
