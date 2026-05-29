# src/core/json_storage.py

import os
import json


class JSONStorage:

    BASE_PATH = "data/processed"

    @staticmethod
    def _get_project_path(project_id):
        return os.path.join(JSONStorage.BASE_PATH, project_id)

    @staticmethod
    def save(project_id, filename, new_records):
        """
        Appends new records to existing JSON file.
        If file does not exist, it creates it.
        """

        project_path = JSONStorage._get_project_path(project_id)
        os.makedirs(project_path, exist_ok=True)

        file_path = os.path.join(project_path, filename)

        # Load existing records if present
        existing_records = []

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    existing_records = json.load(f)
                    if not isinstance(existing_records, list):
                        existing_records = []
                except json.JSONDecodeError:
                    existing_records = []

        # Append and de-duplicate exact record payloads so repeated uploads
        # do not silently pollute downstream inference and retrieval.
        combined_records = JSONStorage._dedupe_records(existing_records + new_records)

        # Write back combined records
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_records, f, indent=4)

        print(f"{len(new_records)} record(s) appended. Total records now: {len(combined_records)}")

    @staticmethod
    def load(project_id, filename):

        file_path = os.path.join(
            JSONStorage.BASE_PATH,
            project_id,
            filename
        )

        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    @staticmethod
    def _dedupe_records(records):
        unique_records = []
        seen = set()

        for record in records:
            try:
                key = json.dumps(record, sort_keys=True, ensure_ascii=False)
            except TypeError:
                key = str(record)

            if key in seen:
                continue

            seen.add(key)
            unique_records.append(record)

        return unique_records
