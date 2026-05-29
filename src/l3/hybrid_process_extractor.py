from sentence_transformers import SentenceTransformer, util


class HybridProcessExtractor:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.templates = [
            "appointing party responsibilities",
            "information container management",
            "common data environment workflow",
            "BIM execution plan preparation",
            "information delivery planning",
            "project information management"
        ]

        self.template_embeddings = self.model.encode(self.templates)

    def extract(self, clauses):

        processes = []

        for clause in clauses:

            if self._is_process_rule(clause):

                processes.append({
                    "rule_type": "ProcessRule",
                    "text": clause
                })

        return processes

    def _is_process_rule(self, text):

        text_lower = text.lower()

        # remove table-of-contents garbage
        if "edition" in text_lower:
            return False

        if "page" in text_lower:
            return False

        if len(text_lower) < 40:
            return False

        emb = self.model.encode(text)

        score = util.cos_sim(emb, self.template_embeddings)

        return score.max() > 0.55