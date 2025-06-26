# ispec/api/routes/projects.py
from fastapi import APIRouter
from api.models.project import Project

# import the necessary db objects
from ispec.db.connect import get_connection
from ispec.db.crud import Person, TableCRUD, Project, ProjectPerson, ProjectComment, ProjectNote, LetterOfS


router = APIRouter()

@router.get("/", response_model=list[Project])
def get_all_projects(): # example
    return [
        Project(id=1, title="Mock A", description="...", status="Planning"),
        Project(id=2, title="Mock B", description="...", status="Ongoing"),
    ]

