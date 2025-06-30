from typing import Type, Callable
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException


from ispec.db.connect import get_session
from ispec.db.models import Person, Project, ProjectComment
from ispec.db.crud import PersonCRUD, ProjectCRUD, ProjectCommentCRUD
from ispec.api.models.modelmaker import get_models, make_pydantic_model_from_sqlalchemy

models = get_models()
# ProjectRead = models["ProjectRead"]
# ProjectUpdate = models["ProjectUpdate"]

router = APIRouter()


# @router.get("/projects/{project_id}", response_model=ProjectRead)
# def read_project(project_id: int, db: Session = Depends(get_session)):
#     project_crud = ProjectCRUD()
#     return project_crud.get(db, project_id)


# @router.put("/projects/{project_id}", response_model=ProjectRead)
# def update_project(
#     project_id: int, payload: ProjectUpdate, db: Session = Depends(get_session)
# ):
#     project_crud = ProjectCRUD()
#     obj = project_crud.get(db, project_id)
#     for field, value in payload.model_dump(exclude_unset=True).items():
#         setattr(obj, field, value)
#     db.commit()
#     db.refresh(obj)
#     return obj


def generate_crud_router(
    model,
    crud_class,
    *,
    prefix: str,
    tag: str,
    # strip_prefix: str = "",
    exclude_fields: set[str] = {"id"},
    optional_all: bool = False,
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=[tag])
    crud = crud_class()

    # Generate models
    ReadModel = make_pydantic_model_from_sqlalchemy(
        model,
        name_suffix="Read",
        # strip_prefix=strip_prefix,
    )
    CreateModel = make_pydantic_model_from_sqlalchemy(
        model,
        name_suffix="Create",
        # strip_prefix=strip_prefix,
        exclude_fields={*exclude_fields, "CreationTS", "ModificationTS"},
        optional_all=optional_all,
        # exclude_fields = {""}
    )

    @router.get("/{item_id}", response_model=ReadModel)
    def get_item(item_id: int, db: Session = Depends(get_session)):
        obj = crud.get(db, item_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        return obj

    @router.post(
        "/",
        response_model=ReadModel,
        summary=f"Create new {tag}",
        description="Create a new item. Required fields are marked with * in the request body below.",
    )
    def create_item(payload: CreateModel, db: Session = Depends(get_session)):
        obj = crud.create(db, payload.model_dump())
        if obj is None:
            raise HTTPException(status_code=409, detail=f"{tag} already exists")
        return obj

    @router.put("/{item_id}", response_model=ReadModel)
    def update_item(
        item_id: int, payload: CreateModel, db: Session = Depends(get_session)
    ):
        obj = crud.get(db, item_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(obj, field, value)
        db.commit()
        db.refresh(obj)
        return obj

    @router.delete("/{item_id}")
    def delete_item(item_id: int, db: Session = Depends(get_session)):
        success = crud.delete(db, item_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        return {"status": "deleted", "id": item_id}

    return router


# person_router = generate_crud_router(
#     model=Person,
#     crud_class=PersonCRUD,
#     prefix="/people",
#     tag="Person",
#     strip_prefix="ppl_",
#     exclude_fields={"id", "ppl_AddedBy"},
#     optional_all=False,  # Change to True if you want POST to accept partial fields
# )

router.include_router(
    generate_crud_router(
        model=Person,
        crud_class=PersonCRUD,
        prefix="/people",
        tag="Person",
        # strip_prefix="ppl_",
        exclude_fields={"id", "ppl_AddedBy"},
    )
)

router.include_router(
    generate_crud_router(
        model=Project,
        crud_class=ProjectCRUD,
        prefix="/projects",
        tag="Project",
        # strip_prefix="prj_",
        exclude_fields={"id"},
    )
)


router.include_router(
    generate_crud_router(
        model=ProjectComment,
        crud_class=ProjectCommentCRUD,
        prefix="/project_comment",
        tag="ProjectComment",
        # strip_prefix="com_",
        exclude_fields={"id"},
    )
)


@router.get("/status")
def status():
    return {"ok": True}
