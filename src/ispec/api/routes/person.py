# api/routes/person.py
# somewhere import get_db
# from FastAPI import Depends
# this is exmaple file, we likely put all the routes together
@router.post("/person/", response_model=PersonOut)
def create_person(person: PersonCreate, db: Session = Depends(get_db)):
    created = person_crud.create(db, person.dict())
    if not created:
        raise HTTPException(status_code=400, detail="Insert failed")
    return created


@router.get("/person/{id}", response_model=PersonOut)
def read_person(id: int, db: Session = Depends(get_db)):
    person = person_crud.get_by_id(db, id)
    if not person:
        raise HTTPException(status_code=404, detail="Not found")
    return person
