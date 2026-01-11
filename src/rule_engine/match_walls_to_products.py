import json
from pathlib import Path


IFC_WALLS_PATH = "data/output/ifc_walls.json"
PRODUCTS_PATH = "data/output/products.json"
RULES_PATH = "data/output/rules.json"
OUTPUT_PATH = "data/output/mismatches.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fire_rating_to_minutes(rating):
    """
    Converts EI60 → 60, EI120 → 120, etc.
    """
    if rating is None:
        return None
    if isinstance(rating, str) and rating.startswith("EI"):
        return int(rating.replace("EI", ""))
    return None


def is_external_wall(wall):
    return wall.get("psets", {}).get("Pset_WallCommon", {}).get("IsExternal") is True


def find_matching_product(wall, products, required_fire_minutes):
    for product in products:
        product_fire = fire_rating_to_minutes(product["fire_rating"])
        if product_fire is None:
            continue

        if (
            product["application"] == "External"
            and product_fire >= required_fire_minutes
        ):
            return product

    return None


def main():
    walls = load_json(IFC_WALLS_PATH)
    products = load_json(PRODUCTS_PATH)
    rules = load_json(RULES_PATH)

    # Find fire rule (simple MVP logic)
    fire_rule = next(
        r for r in rules if r["property"] == "FireRating"
    )
    required_fire_minutes = int(fire_rule["value"])

    mismatches = []

    for wall in walls:
        if not is_external_wall(wall):
            continue

        matched_product = find_matching_product(
            wall, products, required_fire_minutes
        )

        if matched_product is None:
            mismatches.append({
                "wall_id": wall["global_id"],
                "wall_name": wall["name"],
                "issue": "No compliant product found",
                "required_fire_rating": f"EI{required_fire_minutes}",
                "wall_fire_rating": wall.get("fire_rating"),
            })

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mismatches, f, indent=2)

    print(f"PHASE 3 COMPLETE — {len(mismatches)} mismatches found")


if __name__ == "__main__":
    main()
