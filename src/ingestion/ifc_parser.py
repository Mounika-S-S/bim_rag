import ifcopenshell
from src.core.schema import create_layer_record


class IFCParser:

    def parse_ifc(self, file_path):

        model = ifcopenshell.open(file_path)
        records = []

        for element in model.by_type("IfcElement"):

            record_id = element.GlobalId
            entity_type = element.is_a()

            properties = {}

            # Basic properties
            properties["Name"] = element.Name
            properties["ObjectType"] = getattr(element, "ObjectType", None)

            # Extract Property Sets
            if hasattr(element, "IsDefinedBy"):
                for rel in element.IsDefinedBy:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        pset = rel.RelatingPropertyDefinition
                        if pset.is_a("IfcPropertySet"):
                            for prop in pset.HasProperties:
                                if hasattr(prop, "NominalValue") and prop.NominalValue:
                                    properties[prop.Name] = prop.NominalValue.wrappedValue

            record = create_layer_record(
                record_id=record_id,
                entity_type=entity_type,
                layer="L1",
                category="General",
                properties=properties
            )

            records.append(record)

        return records