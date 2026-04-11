import ifcopenshell
from src.core.schema import create_layer_record

# Map IFC entity types → normalized string used across all layers
ENTITY_TYPE_MAP = {
    "IfcWall": "wall", "IfcWallStandardCase": "wall",
    "IfcSlab": "slab", "IfcRoof": "roof",
    "IfcBeam": "beam", "IfcColumn": "column",
    "IfcStair": "stair", "IfcStairFlight": "stair",
    "IfcDoor": "door", "IfcWindow": "window",
    "IfcFoundation": "foundation", "IfcFooting": "foundation",
    "IfcPile": "pile", "IfcRamp": "ramp",
    "IfcCovering": "covering", "IfcCurtainWall": "curtain_wall",
    "IfcPlate": "plate", "IfcMember": "member",
}


class IFCParser:

    def parse_ifc(self, file_path):
        model = ifcopenshell.open(file_path)
        records = []

        for element in model.by_type("IfcElement"):
            record_id = element.GlobalId
            entity_type = element.is_a()
            element_type_normalized = ENTITY_TYPE_MAP.get(entity_type, entity_type.lower().replace("ifc", ""))

            properties = {
                "Name": element.Name,
                "ObjectType": getattr(element, "ObjectType", None),
            }

            # -- 1. Instance-level Property Sets (IsDefinedBy) --
            if hasattr(element, "IsDefinedBy"):
                for rel in element.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        pset = rel.RelatingPropertyDefinition
                        if pset.is_a("IfcPropertySet"):
                            for prop in pset.HasProperties:
                                if hasattr(prop, "NominalValue") and prop.NominalValue:
                                    properties[prop.Name] = prop.NominalValue.wrappedValue
                        # Quantity Sets (area, volume, length)
                        elif pset.is_a("IfcElementQuantity"):
                            for qty in pset.Quantities:
                                qval = None
                                if hasattr(qty, "LengthValue"):
                                    qval = qty.LengthValue
                                elif hasattr(qty, "AreaValue"):
                                    qval = qty.AreaValue
                                elif hasattr(qty, "VolumeValue"):
                                    qval = qty.VolumeValue
                                elif hasattr(qty, "WeightValue"):
                                    qval = qty.WeightValue
                                if qval is not None:
                                    properties[f"Qty_{qty.Name}"] = round(qval, 4)

            # -- 2. Type-level Property Sets (IsTypedBy) --
            if hasattr(element, "IsTypedBy"):
                for rel in element.IsTypedBy:
                    element_type_obj = rel.RelatingType
                    if hasattr(element_type_obj, "HasPropertySets") and element_type_obj.HasPropertySets:
                        for pset in element_type_obj.HasPropertySets:
                            if pset.is_a("IfcPropertySet"):
                                for prop in pset.HasProperties:
                                    if hasattr(prop, "NominalValue") and prop.NominalValue:
                                        # Don't overwrite instance-level values
                                        if prop.Name not in properties:
                                            properties[prop.Name] = prop.NominalValue.wrappedValue

            # -- 3. Material Layer Sets --
            material_info = self._extract_material(element)
            if material_info:
                properties.update(material_info)

            # Normalize known property names for cross-layer alignment
            properties = self._normalize_property_names(properties)

            record = create_layer_record(
                record_id=record_id,
                entity_type=entity_type,
                layer="L1",
                category="IFCElement",
                properties=properties,
                element_type_normalized=element_type_normalized,
            )
            records.append(record)

        return records

    def _extract_material(self, element):
        """Extract material name and total thickness from material layer sets."""
        material_info = {}
        try:
            if hasattr(element, "HasAssociations"):
                for rel in element.HasAssociations:
                    if rel.is_a("IfcRelAssociatesMaterial"):
                        mat = rel.RelatingMaterial
                        if mat.is_a("IfcMaterialLayerSetUsage"):
                            layer_set = mat.ForLayerSet
                            layers = layer_set.MaterialLayers
                            names = []
                            total_thickness = 0.0
                            for layer in layers:
                                if layer.Material and layer.Material.Name:
                                    names.append(layer.Material.Name)
                                if layer.LayerThickness:
                                    total_thickness += layer.LayerThickness
                            if names:
                                material_info["Material"] = ", ".join(names)
                            if total_thickness:
                                material_info["Thickness_mm"] = round(total_thickness * 1000, 1)
                        elif mat.is_a("IfcMaterial"):
                            material_info["Material"] = mat.Name
                        elif mat.is_a("IfcMaterialList"):
                            material_info["Material"] = ", ".join(
                                m.Name for m in mat.Materials if m.Name
                            )
        except Exception:
            pass
        return material_info

    def _normalize_property_names(self, props):
        """Map common IFC property names to canonical cross-layer names."""
        alias = {
            "FireRating": "FireRating_min",
            "Fire Rating": "FireRating_min",
            "fire_rating": "FireRating_min",
            "Thickness": "Thickness_mm",
            "OverallThickness": "Thickness_mm",
            "Width": "Width_mm",
            "Height": "Height_mm",
            "OverallHeight": "Height_mm",
            "OverallWidth": "Width_mm",
            "LoadBearing": "IsLoadBearing",
            "IsExternal": "IsExternal",
        }
        normalized = {}
        for k, v in props.items():
            canonical = alias.get(k, k)
            normalized[canonical] = v
        return normalized