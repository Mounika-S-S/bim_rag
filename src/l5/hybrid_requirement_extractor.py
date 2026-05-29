from sentence_transformers import SentenceTransformer, util


class HybridRequirementExtractor:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.templates = [
            "labour requirement",
            "material supply requirement",
            "construction rate requirement",
            "technical staff requirement",
            "equipment requirement",
            "project execution requirement"
        ]

        self.template_embeddings = self.model.encode(self.templates)

    def extract(self, clauses):

        requirements = []

        for clause in clauses:

            if self._is_requirement(clause):

                requirements.append({
                    "rule_type": "Requirement",
                    "text": clause
                })

        return requirements

    def _is_requirement(self, text):

        emb = self.model.encode(text)

        score = util.cos_sim(emb, self.template_embeddings)

        return score.max() > 0.55