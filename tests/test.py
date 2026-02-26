from src.ingestion.ifc_parser import IFCParser
from src.ingestion.product_parser import ProductParser
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.regulation_parser import RegulationParser
pipeline = IngestionPipeline()

# L1
ifc_parser = IFCParser()
l1_elements = ifc_parser.parse_json("data/raw/ifc/sample_ifc.json")
pipeline.ingest_elements(l1_elements)

# L2
product_parser = ProductParser()
l2_elements = product_parser.parse_json("data/raw/products/sample_products.json")
pipeline.ingest_elements(l2_elements)

# L4
reg_parser = RegulationParser()
l4_elements = reg_parser.parse_json("data/raw/regulations/sample_regulations.json")
pipeline.ingest_elements(l4_elements)

result = pipeline.vector_store.query("minimum fire resistance required for walls")
print(result)