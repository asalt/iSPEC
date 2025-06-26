# schemas.py
from pydantic import BaseModel

class PersonCreate(BaseModel):
    ppl_Name_First: str
    ppl_Name_Last: str

class PersonOut(PersonCreate):
    id: int

    class Config:
        orm_mode = True  # <-- enables use of SQLAlchemy ORM object directly

