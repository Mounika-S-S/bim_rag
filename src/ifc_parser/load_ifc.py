import ifcopenshell
from get_elements import get_walls
from extract_basic import extract_basic_info
from extract_psets import extract_psets
from extract_wall_data import extract_wall_data

def load_ifc_model(ifc_path: str):
    """
    Loads an IFC file and returns the model object.
    """
    model = ifcopenshell.open(ifc_path)
    return model


if __name__ == "__main__":
    model = load_ifc_model("data/ifc/sample.ifc")
    print("IFC schema:", model.schema)
    print("Total IfcProduct elements:", len(model.by_type("IfcProduct")))
    walls = get_walls(model)
    print("Total walls:", len(walls))
    sample_wall = walls[0]
    print(extract_basic_info(sample_wall))
    psets = extract_psets(sample_wall)
    print(psets.keys())
    wall_data = extract_wall_data(sample_wall)
    print(wall_data)