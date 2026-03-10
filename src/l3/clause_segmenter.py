import re


class ClauseSegmenter:

    def segment(self, text):

        clauses = re.split(r"(?=\d+\.\d+)", text)

        cleaned = []

        for c in clauses:

            c = c.strip()

            if len(c) < 40:
                continue

            # skip tables
            if "|" in c:
                continue

            # skip list-of-tables
            if "table" in c.lower():
                continue

            cleaned.append(c)

        return cleaned