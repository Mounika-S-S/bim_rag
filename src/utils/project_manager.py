import os


class ProjectManager:

    BASE_PATH = "vector_store"

    @classmethod
    def initialize_base(cls):
        os.makedirs(cls.BASE_PATH, exist_ok=True)

    @classmethod
    def list_projects(cls):
        cls.initialize_base()
        return [
            name for name in os.listdir(cls.BASE_PATH)
            if os.path.isdir(os.path.join(cls.BASE_PATH, name))
        ]

    @classmethod
    def project_exists(cls, project_id: str):
        return os.path.exists(os.path.join(cls.BASE_PATH, project_id))

    @classmethod
    def create_project(cls, project_id: str):
        os.makedirs(os.path.join(cls.BASE_PATH, project_id), exist_ok=True)

    @classmethod
    def delete_project(cls, project_id: str):
        import shutil
        shutil.rmtree(os.path.join(cls.BASE_PATH, project_id), ignore_errors=True)