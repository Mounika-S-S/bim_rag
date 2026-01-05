import json
from .load_ifc import load_ifc_model
from .get_elements import get_walls
from .extract_wall_data import extract_wall_data

model = load_ifc_model("data/ifc/sample.ifc")
walls = get_walls(model)

wall_data = [extract_wall_data(w) for w in walls]

with open("data/output/ifc_walls.json", "w") as f:
    json.dump(wall_data, f, indent=2)

print("Saved", len(wall_data), "walls")
