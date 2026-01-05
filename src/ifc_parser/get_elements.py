def get_walls(model):
    """
    Returns all wall elements from the IFC model.
    """
    walls = []
    walls.extend(model.by_type("IfcWall"))
    walls.extend(model.by_type("IfcWallStandardCase"))
    return walls
