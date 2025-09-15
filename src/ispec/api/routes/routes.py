from typing import Type, Callable
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, Query


from ispec.db.connect import get_session
from ispec.db.models import Person, Project, ProjectComment
from ispec.db.crud import PersonCRUD, ProjectCRUD, ProjectCommentCRUD

from ispec.api.routes.utils.ui_meta import ui_from_column  # used inside schema builder
from ispec.api.routes.schema import build_form_schema

from ispec.api.models.modelmaker import get_models, make_pydantic_model_from_sqlalchemy

# models = get_models() 
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



ROUTE_PREFIX_BY_TABLE: dict[str, str] = {}


def generate_crud_router(
    model,
    crud_class,
    *,
    prefix: str,
    tag: str,
    exclude_fields: set[str] = {"id"},
    create_exclude_fields: set[str] | None = None,
    optional_all: bool = False,
    route_prefix_by_table: dict[str, str] | None = None,
) -> APIRouter:
    """
    Create a FastAPI router providing CRUD and metadata endpoints for a SQLAlchemy model.

    This function dynamically generates a FastAPI `APIRouter` for the given
    SQLAlchemy model and associated CRUD class. It auto-constructs Pydantic
    models for reading and creating objects, and wires them into standard
    REST-style endpoints:

      - `GET /schema` – Return the Pydantic create model's JSON schema,
        annotated with UI metadata from `ui_from_column()` (or from
        model-level metadata such as `col.info["group"]`).
      - `GET /{item_id}` – Retrieve a single object by primary key.
      - `POST /` – Create a new object; required fields marked in schema.
      - `PUT /{item_id}` – Update an existing object (partial updates supported).
      - `DELETE /{item_id}` – Delete an object by primary key.
      - `GET /options` – List available options for select inputs (supports query filtering, ID filtering, and exclusions).
      - `GET /options/{field}` – List options for a specific relationship
        field (WIP; resolves related model dynamically).

    The generated router uses `crud_class` for database operations and
    `make_pydantic_model_from_sqlalchemy()` to derive request/response models.

    Parameters
    ----------
    model : SQLAlchemy declarative model class
        The ORM model to generate endpoints for.
    crud_class : type
        A class implementing CRUD methods (`get`, `create`, `delete`, `list_options`).
    prefix : str
        URL path prefix for all routes in the router.
    tag : str
        Tag name for FastAPI's OpenAPI documentation grouping.
    exclude_fields : set[str], default={"id"}
        Fields to exclude from *all* generated Pydantic models.
    create_exclude_fields : set[str] | None, default=None
        Fields to exclude from the create/update Pydantic model only.
    optional_all : bool, default=False
        If True, marks all fields in the create/update model as optional.

    route_prefix_by_table : dict[str, str] | None, default=None
        Mapping of table names to route prefixes. If not provided, a module-level
        global registry is used. This is primarily useful for testing to avoid
        mutating global state.

    Returns
    -------
    APIRouter
        A FastAPI router with the generated CRUD and metadata endpoints.

    Notes
    -----
    - This function relies on `ui_from_column()` to inject UI-specific hints
      into the `/schema` output, which can be used by a frontend to render
      forms automatically.
    - Foreign key fields in the create model will produce async-select UI
      components via `/options` endpoints.
    - The `/options/{field}` endpoint is work-in-progress and may require
      a CRUD registry for related models.
    - This function is designed for extensibility — you can add extra
      endpoints or tweak schema generation after calling it.
    """


    create_exclude_fields = create_exclude_fields or set()
    router = APIRouter(prefix=prefix, tags=[tag])
    crud = crud_class()

    prefix_map = route_prefix_by_table if route_prefix_by_table is not None else ROUTE_PREFIX_BY_TABLE

    # register prefix for FK resolution (e.g., "person" -> "/people")
    prefix_map[model.__table__.name] = prefix

    def route_prefix_for_table(table: str) -> str:
        return prefix_map.get(table, f"/{table}")


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
        exclude_fields={*exclude_fields, *create_exclude_fields},
        optional_all=optional_all,
        # exclude_fields = {""}
    )

    # this is for frontend UI form rendering
    @router.get("/schema")
    def get_schema():
        return build_form_schema(model, CreateModel, route_prefix_for_table=route_prefix_for_table)

        # this is now in build_form_schema
        # schema = CreateModel.model_json_schema()
        # # attach per-field UI hints
        # props = schema.get("properties", {})
        # column_map = {c.name: c for c in model.__table__.columns}  # type: ignore
        # for name, prop in props.items():
        #     col = column_map.get(name)
        #     if col is None:
        #         continue
        #     prop["ui"] = ui_from_column(col)
        #     # carry ordering/grouping if present on SA Column
        #     if grp := (col.info or {}).get("group"):
        #         prop.setdefault("ui", {})["group"] = grp

        # # top-level UI (sections/order) — optional
        # schema["ui"] = {
        #     "order": [c.name for c in model.__table__.columns if c.name in props],
        #     "sections": [],  # you can prefill from model-level metadata if desired
        #     "title": tag,
        # }
        # return schema

    @router.get("/{item_id}", response_model=ReadModel, response_model_exclude_none=True)
    def get_item(item_id: int, db: Session = Depends(get_session)):
        obj = crud.get(db, item_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        return ReadModel.model_validate(obj)

    @router.post(
        "/",
        response_model=ReadModel,
        response_model_exclude_none=True,
        summary=f"Create new {tag}",
        description="Create a new item. Required fields are marked with * in the request body below.",
        status_code=201,
    )
    def create_item(payload: CreateModel, db: Session = Depends(get_session)):
        obj = crud.create(db, payload.model_dump())
        if obj is None:
            raise HTTPException(status_code=409, detail=f"{tag} already exists")
        return ReadModel.model_validate(obj)

    @router.put("/{item_id}", response_model=ReadModel, response_model_exclude_none=True)
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
        return ReadModel.model_validate(obj)

    @router.delete("/{item_id}")
    def delete_item(item_id: int, db: Session = Depends(get_session)):
        success = crud.delete(db, item_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        return {"status": "deleted", "id": item_id}


    @router.get("/options")
    def options(
        q: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
        ids: list[int] | None = Query(default=None),
        exclude_ids: list[int] | None = Query(default=None),
        db: Session = Depends(get_session),
    ):
        return crud.list_options(db, q=q, limit=limit, ids=ids, exclude_ids=exclude_ids)

    # this is WIP
    @router.get("/options/{field}")
    def options_for_field(
        field: str,
        q: str | None = None,
        limit: int = 20,
        db: Session = Depends(get_session),
    ):
        # Resolve the related model via ORM relationship
        from sqlalchemy import inspect as sa_inspect

        mapper = sa_inspect(model)
        rel = mapper.relationships.get(field)
        if not rel:
            raise HTTPException(
                status_code=404, detail=f"No relationship named '{field}'"
            )
        target_cls = rel.mapper.class_
        # Use the target's CRUD (you can keep a registry {Model: CRUD})
        target_crud = (
            crud.__class__()
        )  # if your CRUDs share a base, use a registry instead
        target_crud.model = target_cls
        return target_crud.list_options(db, q=q, limit=limit)

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


# the crud_classes have custom methods for providing list_schema
# UI components guide
# other things

# ========================= Person ==============================
router.include_router(
    generate_crud_router(
        model=Person,
        crud_class=PersonCRUD,
        prefix="/people",
        tag="Person",
        # strip_prefix="ppl_",
        exclude_fields={
            "id",
        },
        create_exclude_fields={"ppl_CreationTS", "ppl_ModificationTS"},
    )
)


# ========================= Project ==============================
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


# ========================= ProjectComment ==============================
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
