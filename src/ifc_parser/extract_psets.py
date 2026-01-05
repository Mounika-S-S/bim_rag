def extract_psets(element):
    """
    Extracts all property sets from an IFC element.
    Returns a dictionary: {PsetName: {PropertyName: value}}
    """
    psets = {}

    if not hasattr(element, "IsDefinedBy"):
        return psets

    for rel in element.IsDefinedBy:
        if rel.is_a("IfcRelDefinesByProperties"):
            prop_set = rel.RelatingPropertyDefinition

            if prop_set.is_a("IfcPropertySet"):
                props = {}
                for prop in prop_set.HasProperties:
                    if prop.is_a("IfcPropertySingleValue"):
                        value = prop.NominalValue
                        props[prop.Name] = (
                            value.wrappedValue if value else None
                        )
                psets[prop_set.Name] = props

    return psets
