from src.l4.clause_segmenter import ClauseSegmenter
from src.l4.rules_extractor import RulesExtractor


class L4Pipeline:

    def __init__(self):
        self.segmenter = ClauseSegmenter()
        self.extractor = RulesExtractor()

    # -------------------------------------

    def parse(self, pdf_path: str):

        clauses = self.segmenter.segment(pdf_path)

        structured_rules = self.extractor.extract(clauses)

        return structured_rules