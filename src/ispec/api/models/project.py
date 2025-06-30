# api/models/project.py
# this is an example file and wil likely be removed, supplanted by modelmaker.py
from pydantic import BaseModel
from typing import Optional


class ProjectBase(BaseModel):
    id: int
    title: str
    description: Optional[str]
    status: str


class Project(ProjectBase):
    id: int

    class Config:
        orm_mode = True


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
