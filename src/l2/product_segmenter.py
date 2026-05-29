import re


class ProductSegmenter:

    def segment(self, text):

        sentences = re.split(r"(?<=[.!?])\s+", text)

        results = []

        for s in sentences:

            s = s.strip()

            if len(s) < 20:
                continue

            # skip table lines
            if "|" in s:
                continue

            if re.search(r"\d+\s*\|\s*", s):
                continue

            results.append(s)

        return results