from sentence_transformers import SentenceTransformer, util
import re


class HybridProductExtractor:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.templates = [
            "product name",
            "manufacturer name",
            "model number",
            "concrete grade",
            "steel grade",
            "compressive strength",
            "beam length",
            "beam width",
            "beam depth",
            "fire rating"
        ]

        self.template_embeddings = self.model.encode(self.templates)

        self.number_pattern = re.compile(r"\d+(\.\d+)?")

    def extract(self, sentences):

        properties = []

        for s in sentences:

            if self._is_property(s):

                value = None

                num = self.number_pattern.search(s)

                if num:
                    value = float(num.group())

                properties.append({
                    "property_text": s,
                    "numeric_value": value
                })

        return properties

    def _is_property(self, text):

        emb = self.model.encode(text)

        scores = util.cos_sim(emb, self.template_embeddings)

        return scores.max() > 0.55