import os
import json


class JSONStorage:

    BASE_PATH = "data/processed"

    @classmethod
    def save(cls, project_id, filename, data):
        project_path = os.path.join(cls.BASE_PATH, project_id)
        os.makedirs(project_path, exist_ok=True)

        file_path = os.path.join(project_path, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return file_path

    @classmethod
    def load(cls, project_id, filename):
        file_path = os.path.join(cls.BASE_PATH, project_id, filename)

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)