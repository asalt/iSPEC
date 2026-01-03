import os
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import or_
from sqlalchemy.orm import Session
from ispec.db.connect import get_session_dep
from typing import Type, Callable
from ispec.db.models import (
    Person,
    Project,
    ProjectComment,
    ProjectPerson,
    E2G,
    Experiment,
    ExperimentRun,
    Job,
    PSM,
    MSRawFile,
    LetterOfSupport,
)
from ispec.db.crud import (
    CRUDBase,
    PersonCRUD,
    ProjectCRUD,
    ProjectCommentCRUD,
    ProjectPersonCRUD,
    E2GCRUD,
    ExperimentCRUD,
    ExperimentRunCRUD,
    JobCRUD,
    PSMCRUD,
    MSRawFileCRUD,
    LetterOfSupportCRUD,
)

from ispec.api.routes.schema import build_form_schema

from ispec.api.models.modelmaker import make_pydantic_model_from_sqlalchemy

router = APIRouter()

# Pydantic read models used by some convenience endpoints
E2GRead = make_pydantic_model_from_sqlalchemy(E2G, name_suffix="Read")
JobRead = make_pydantic_model_from_sqlalchemy(Job, name_suffix="Read")

_ALL_RESOURCES: set[str] = {
    "people",
    "projects",
    "experiments",
    "experiment_runs",
    "experiment_to_gene",
    "jobs",
    "psms",
    "msraw_files",
    "project_comment",
    "project_person",
    "letter_of_support",
}

_DEFAULT_RESOURCES: set[str] = {"projects", "people", "project_comment"}


def _parse_csv_set(raw: str) -> set[str]:
    return {part.strip().lstrip("/").lower() for part in raw.split(",") if part.strip()}


def get_exposed_resources() -> set[str]:
    """Return the set of enabled API resource prefixes.

    Configure via ``ISPEC_API_RESOURCES`` (comma separated). Use ``all`` or ``*``
    to expose everything that is wired into this router module.
    """

    raw = os.getenv("ISPEC_API_RESOURCES")
    if not raw:
        return set(_DEFAULT_RESOURCES)

    resources = _parse_csv_set(raw)
    if not resources:
        return set()

    if "all" in resources or "*" in resources:
        return set(_ALL_RESOURCES)

    return resources


EXPOSED_RESOURCES = get_exposed_resources()


def _enabled(resource: str) -> bool:
    return resource.lstrip("/").lower() in EXPOSED_RESOURCES


# ----- shared list helpers --------------------------------------------------


def _parse_order_part(part: str) -> tuple[str, str] | None:
    raw = (part or "").strip()
    if not raw:
        return None

    direction = "asc"
    if raw[0] in "+-":
        if raw[0] == "-":
            direction = "desc"
        raw = raw[1:].strip()

    if ":" in raw:
        field, dir_part = raw.split(":", 1)
        dir_part = dir_part.strip().lower()
        if dir_part in {"desc", "d"}:
            direction = "desc"
        elif dir_part in {"asc", "a"}:
            direction = "asc"
        raw = field.strip()

    lower = raw.lower()
    if lower.endswith("_desc"):
        raw = raw[:-5].strip()
        direction = "desc"
    elif lower.endswith("_asc"):
        raw = raw[:-4].strip()
        direction = "asc"

    if not raw:
        return None
    return raw, direction


def _apply_ordering(query, model, order: str | None):
    order = (order or "").strip()
    if not order:
        return query.order_by(getattr(model, "id").asc())

    columns = set(getattr(model.__table__, "columns").keys())  # type: ignore[attr-defined]
    parts = [p.strip() for p in order.split(",") if p.strip()]
    order_by = []
    for part in parts:
        parsed = _parse_order_part(part)
        if not parsed:
            continue
        field, direction = parsed
        if field not in columns:
            continue
        expr = getattr(model, field)
        order_by.append(expr.desc() if direction == "desc" else expr.asc())

    if not order_by:
        return query.order_by(getattr(model, "id").asc())

    return query.order_by(*order_by)


# The router previously relied on a module level ROUTE_PREFIX_BY_TABLE for
# resolving foreign-key option endpoints.  Global state makes isolated testing
# difficult, so the mapping is now provided via dependency injection.  Each
# call to ``generate_crud_router`` accepts a mapping which is shared across
# routers when needed.


def _add_schema_endpoint(
    router: APIRouter,
    model,
    create_model,
    *,
    route_prefix_for_table: Callable[[str], str],
) -> None:
    """Register the ``/schema`` endpoint on ``router``."""

    @router.get("/schema")
    def get_schema():  # pragma: no cover - trivial wrapper
        return build_form_schema(
            model, create_model, route_prefix_for_table=route_prefix_for_table
        )


def _add_crud_endpoints(
    router: APIRouter,
    crud,
    read_model,
    create_model,
    *,
    tag: str,
) -> None:
    """Attach basic CRUD endpoints to ``router``."""

    @router.get("")
    @router.get("/")
    def list_items(
        response: Response,
        q: str | None = None,
        limit: int = Query(default=50, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        order: str | None = None,
        ids: list[int] | None = Query(default=None),
        exclude_ids: list[int] | None = Query(default=None),
        wrap: bool = Query(default=False, description="Wrap response as {items,total}"),
        db: Session = Depends(get_session_dep),
    ):
        model = crud.model
        query = db.query(model)

        if ids:
            query = query.filter(getattr(model, "id").in_(ids))
        if exclude_ids:
            query = query.filter(~getattr(model, "id").in_(exclude_ids))
        if q:
            q = q.strip()
            q_int: int | None = None
            if q.isdigit():
                try:
                    q_int = int(q)
                except Exception:
                    q_int = None
            try:
                expr = crud.label_expr().ilike(f"%{q}%")
                if q_int is not None:
                    query = query.filter(or_(getattr(model, "id") == q_int, expr))
                else:
                    query = query.filter(expr)
            except Exception:
                if q_int is not None:
                    query = query.filter(getattr(model, "id") == q_int)

        query = _apply_ordering(query, model, order)

        # compute total before pagination
        try:
            total = query.order_by(None).count()
        except Exception:
            # fallback if order_by(None) unsupported
            total = query.count()

        rows = query.offset(offset).limit(limit).all()
        payload = [read_model.model_validate(r).model_dump() for r in rows]
        # attach total via header for simple lists
        try:
            if response is not None:
                response.headers["X-Total-Count"] = str(total)
        except Exception:
            pass
        if wrap:
            return {"items": payload, "total": total}
        return payload

    @router.get("/{item_id}", response_model=read_model, response_model_exclude_none=True)
    def get_item(item_id: int, db: Session = Depends(get_session_dep)):
        obj = crud.get(db, item_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        return read_model.model_validate(obj).model_dump()

    @router.post(
        "",
        response_model=read_model,
        response_model_exclude_none=True,
        summary=f"Create new {tag}",
        description="Create a new item. Required fields are marked with * in the request body below.",
        status_code=201,
    )
    @router.post(
        "/",
        response_model=read_model,
        response_model_exclude_none=True,
        summary=f"Create new {tag}",
        description="Create a new item. Required fields are marked with * in the request body below.",
        status_code=201,
    )
    def create_item(payload: create_model, db: Session = Depends(get_session_dep)):
        # Use exclude_unset so SQLAlchemy/Python defaults can still apply when
        # the client omits optional fields (instead of forcing NULL).
        obj = crud.create(db, payload.model_dump(exclude_unset=True))
        if obj is None:
            raise HTTPException(status_code=409, detail=f"{tag} already exists")
        return read_model.model_validate(obj).model_dump()

    @router.put("/{item_id}", response_model=read_model, response_model_exclude_none=True)
    def update_item(item_id: int, payload: create_model, db: Session = Depends(get_session_dep)):
        obj = crud.get(db, item_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(obj, field, value)
        db.commit()
        db.refresh(obj)
        return read_model.model_validate(obj).model_dump()

    @router.delete("/{item_id}")
    def delete_item(item_id: int, db: Session = Depends(get_session_dep)):
        success = crud.delete(db, item_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        return {"status": "deleted", "id": item_id}


def _add_options_endpoints(router: APIRouter, crud, *, model) -> None:
    """Attach ``/options`` endpoints for async select components."""

    @router.get("/options")
    def options(
        q: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
        ids: list[int] | None = Query(default=None),
        exclude_ids: list[int] | None = Query(default=None),
        db: Session = Depends(get_session_dep),
    ):
        return crud.list_options(
            db, q=q, limit=limit, ids=ids, exclude_ids=exclude_ids
        )

    @router.get("/options/{field}")
    def options_for_field(
        field: str,
        q: str | None = None,
        limit: int = 20,
        db: Session = Depends(get_session_dep),
    ):
        from sqlalchemy import inspect as sa_inspect

        mapper = sa_inspect(model)
        rel = mapper.relationships.get(field)
        if not rel:
            raise HTTPException(status_code=404, detail=f"No relationship named '{field}'")
        target_cls = rel.mapper.class_
        crud_class_map = {
            Person: PersonCRUD,
            Project: ProjectCRUD,
            ProjectComment: ProjectCommentCRUD,
            Experiment: ExperimentCRUD,
            ExperimentRun: ExperimentRunCRUD,
        }

        target_crud_cls = crud_class_map.get(target_cls, CRUDBase)
        if target_crud_cls is CRUDBase:
            target_crud = target_crud_cls(target_cls)
        else:
            target_crud = target_crud_cls()
            target_crud.model = target_cls
        return target_crud.list_options(db, q=q, limit=limit)


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
        Fields to exclude from generated Pydantic models. The primary key
        ``id`` is always preserved on read models, even if listed here.
    create_exclude_fields : set[str] | None, default=None
        Fields to exclude from the create/update Pydantic model only.
    optional_all : bool, default=False
        If True, marks all fields in the create/update model as optional.

    route_prefix_by_table : dict[str, str] | None, default=None
        Mapping of table names to route prefixes.  When multiple routers are
        generated the mapping should be shared so that foreign-key fields can
        resolve the appropriate ``/options`` endpoint for related tables.

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


    create_exclude_fields = set(create_exclude_fields or set())
    # Never ask users to supply timestamp mixin fields on create/update.
    create_exclude_fields.update(
        {
            c.name
            for c in model.__table__.columns  # type: ignore[attr-defined]
            if c.name.endswith("_CreationTS") or c.name.endswith("_ModificationTS")
        }
    )
    router = APIRouter(prefix=prefix, tags=[tag])
    crud = crud_class()

    prefix_map = route_prefix_by_table if route_prefix_by_table is not None else {}

    # register prefix for FK resolution (e.g., "person" -> "/people")
    prefix_map[model.__table__.name] = prefix

    def route_prefix_for_table(table: str) -> str:
        return prefix_map.get(table, f"/{table}")

    # Ensure read models include the primary key even when ``exclude_fields``
    # contains "id"; other exclusions still apply.
    read_exclude_fields = {f for f in exclude_fields if f != "id"}
    ReadModel = make_pydantic_model_from_sqlalchemy(
        model,
        name_suffix="Read",
        exclude_fields=read_exclude_fields,
    )
    CreateModel = make_pydantic_model_from_sqlalchemy(
        model,
        name_suffix="Create",
        exclude_fields={*exclude_fields, *create_exclude_fields},
        optional_all=optional_all,
    )

    _add_schema_endpoint(
        router, model, CreateModel, route_prefix_for_table=route_prefix_for_table
    )

    # Model-specific convenience endpoints
    if model is E2G:
        @router.get("/by_run/{run_id}", response_model=list[ReadModel])
        def list_genes_by_run(
            run_id: int,
            q: str | None = None,
            geneidtype: str | None = None,
            limit: int = Query(default=100, ge=1, le=1000),
            offset: int = Query(default=0, ge=0),
            db: Session = Depends(get_session_dep),
        ):
            query = db.query(E2G).filter(E2G.experiment_run_id == run_id)
            if q:
                query = query.filter(E2G.gene.ilike(f"%{q}%"))
            if geneidtype:
                query = query.filter(E2G.geneidtype == geneidtype)
            rows = query.order_by(E2G.id.asc()).limit(limit).offset(offset).all()
            return [ReadModel.model_validate(r).model_dump() for r in rows]

    if model is PSM:
        @router.get("/by_run/{run_id}", response_model=list[ReadModel])
        def list_psms_by_run(
            run_id: int,
            q: str | None = None,
            limit: int = Query(default=200, ge=1, le=5000),
            offset: int = Query(default=0, ge=0),
            db: Session = Depends(get_session_dep),
        ):
            query = db.query(PSM).filter(PSM.experiment_run_id == run_id)
            if q:
                # lightweight search over common text fields
                query = query.filter(
                    (PSM.peptide.ilike(f"%{q}%")) | (PSM.protein.ilike(f"%{q}%"))
                )
            rows = query.order_by(PSM.id.asc()).limit(limit).offset(offset).all()
            return [ReadModel.model_validate(r).model_dump() for r in rows]

    if model is MSRawFile:
        @router.get("/by_run/{run_id}", response_model=list[ReadModel])
        def list_raw_files_by_run(
            run_id: int,
            q: str | None = None,
            limit: int = Query(default=200, ge=1, le=5000),
            offset: int = Query(default=0, ge=0),
            db: Session = Depends(get_session_dep),
        ):
            query = db.query(MSRawFile).filter(MSRawFile.experiment_run_id == run_id)
            if q:
                query = query.filter(MSRawFile.uri.ilike(f"%{q}%"))
            rows = query.order_by(MSRawFile.id.asc()).limit(limit).offset(offset).all()
            return [ReadModel.model_validate(r).model_dump() for r in rows]

    # Register the options endpoints *before* CRUD handlers so that the
    # ``/options`` and ``/options/{field}`` paths take precedence over the
    # generic ``/{item_id}`` route. Starlette matches routes in definition
    # order and would otherwise treat ``/options`` as a candidate for
    # ``/{item_id}``, resulting in a ``422`` response when the path parameter
    # fails integer validation.
    _add_options_endpoints(router, crud, model=model)
    _add_crud_endpoints(
        router, crud, ReadModel, CreateModel, tag=tag
    )

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
_ROUTE_PREFIX_MAP: dict[str, str] = {}

if _enabled("people"):
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
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= Project ==============================
if _enabled("projects"):
    @router.get("/projects/stats")
    def project_stats(db: Session = Depends(get_session_dep)):
        from sqlalchemy import func

        total = db.query(func.count(Project.id)).scalar() or 0
        current = (
            db.query(func.count(Project.id))
            .filter(Project.prj_Current_FLAG.is_(True))
            .scalar()
            or 0
        )
        to_bill = (
            db.query(func.count(Project.id))
            .filter(Project.prj_Billing_ReadyToBill.is_(True))
            .scalar()
            or 0
        )
        paid = (
            db.query(func.count(Project.id))
            .filter(Project.prj_PaymentReceived.is_(True))
            .scalar()
            or 0
        )
        return {
            "projects_total": int(total),
            "projects_current": int(current),
            "projects_to_bill": int(to_bill),
            "projects_paid": int(paid),
        }

    router.include_router(
        generate_crud_router(
            model=Project,
            crud_class=ProjectCRUD,
            prefix="/projects",
            tag="Project",
            # strip_prefix="prj_",
            exclude_fields={"id"},
            create_exclude_fields={"prj_CreationTS", "prj_ModificationTS"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= Experiment ==============================
if _enabled("experiments"):
    router.include_router(
        generate_crud_router(
            model=Experiment,
            crud_class=ExperimentCRUD,
            prefix="/experiments",
            tag="Experiment",
            exclude_fields={"id"},
            create_exclude_fields={"Experiment_CreationTS", "Experiment_ModificationTS"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= ExperimentRun ==============================
if _enabled("experiment_runs"):
    router.include_router(
        generate_crud_router(
            model=ExperimentRun,
            crud_class=ExperimentRunCRUD,
            prefix="/experiment_runs",
            tag="ExperimentRun",
            exclude_fields={"id"},
            create_exclude_fields={"ExperimentRun_CreationTS", "ExperimentRun_ModificationTS"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )

# ========================= PSM ==============================
if _enabled("psms"):
    router.include_router(
        generate_crud_router(
            model=PSM,
            crud_class=PSMCRUD,
            prefix="/psms",
            tag="PSM",
            exclude_fields={"id"},
            create_exclude_fields={"psm_CreationTS", "psm_ModificationTS"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )

# ========================= MS Raw File ==============================
if _enabled("msraw_files"):
    router.include_router(
        generate_crud_router(
            model=MSRawFile,
            crud_class=MSRawFileCRUD,
            prefix="/msraw_files",
            tag="MSRawFile",
            exclude_fields={"id"},
            create_exclude_fields={"msraw_CreationTS", "msraw_ModificationTS"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= LetterOfSupport ==============================
if _enabled("letter_of_support"):
    los_router = APIRouter(prefix="/letter_of_support", tags=["LetterOfSupport"])
    los_crud = LetterOfSupportCRUD()

    LetterRead = make_pydantic_model_from_sqlalchemy(
        LetterOfSupport,
        name_suffix="Read",
        exclude_fields={"los_LOS_docx", "los_LOS_pdf"},
    )

    @los_router.get("")
    @los_router.get("/")
    def list_letters(
        response: Response,
        q: str | None = None,
        limit: int = Query(default=50, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        order: str | None = None,
        ids: list[int] | None = Query(default=None),
        exclude_ids: list[int] | None = Query(default=None),
        wrap: bool = Query(default=False, description="Wrap response as {items,total}"),
        db: Session = Depends(get_session_dep),
    ):
        model = los_crud.model
        query = db.query(model)

        if ids:
            query = query.filter(getattr(model, "id").in_(ids))
        if exclude_ids:
            query = query.filter(~getattr(model, "id").in_(exclude_ids))
        if q:
            try:
                query = query.filter(los_crud.label_expr().ilike(f"%{q}%"))
            except Exception:
                pass

        if order == "id":
            query = query.order_by(getattr(model, "id").asc())
        else:
            query = query.order_by(getattr(model, "id").asc())

        try:
            total = query.order_by(None).count()
        except Exception:
            total = query.count()

        rows = query.offset(offset).limit(limit).all()
        payload = [LetterRead.model_validate(r).model_dump() for r in rows]
        try:
            if response is not None:
                response.headers["X-Total-Count"] = str(total)
        except Exception:
            pass

        if wrap:
            return {"items": payload, "total": total}
        return payload

    @los_router.get("/{item_id}", response_model=LetterRead, response_model_exclude_none=True)
    def get_letter(item_id: int, db: Session = Depends(get_session_dep)):
        obj = los_crud.get(db, item_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="LetterOfSupport not found")
        return LetterRead.model_validate(obj).model_dump()

    def _download_letter_blob(
        item_id: int,
        *,
        blob_attr: str,
        media_type: str,
        default_ext: str,
        db: Session,
    ) -> Response:
        obj = los_crud.get(db, item_id)
        if obj is None:
            raise HTTPException(status_code=404, detail="LetterOfSupport not found")

        blob = getattr(obj, blob_attr, None)
        if not blob:
            raise HTTPException(status_code=404, detail="File not found")

        base = (getattr(obj, "los_FileName", None) or f"letter_{item_id}").strip() or f"letter_{item_id}"
        filename = base
        lower = filename.lower()
        if not lower.endswith(default_ext):
            filename = f"{filename}{default_ext}"

        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=blob, media_type=media_type, headers=headers)

    @los_router.get("/{item_id}/pdf")
    def download_letter_pdf(item_id: int, db: Session = Depends(get_session_dep)):
        return _download_letter_blob(
            item_id,
            blob_attr="los_LOS_pdf",
            media_type="application/pdf",
            default_ext=".pdf",
            db=db,
        )

    @los_router.get("/{item_id}/docx")
    def download_letter_docx(item_id: int, db: Session = Depends(get_session_dep)):
        return _download_letter_blob(
            item_id,
            blob_attr="los_LOS_docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            default_ext=".docx",
            db=db,
        )

    router.include_router(los_router)
    _ROUTE_PREFIX_MAP[LetterOfSupport.__table__.name] = "/letter_of_support"


# ========================= ProjectComment ==============================
if _enabled("project_comment"):
    router.include_router(
        generate_crud_router(
            model=ProjectComment,
            crud_class=ProjectCommentCRUD,
            prefix="/project_comment",
            tag="ProjectComment",
            # strip_prefix="com_",
            exclude_fields={"id"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= ProjectPerson ==============================
if _enabled("project_person"):
    router.include_router(
        generate_crud_router(
            model=ProjectPerson,
            crud_class=ProjectPersonCRUD,
            prefix="/project_person",
            tag="ProjectPerson",
            exclude_fields={"id"},
            create_exclude_fields={"projper_CreationTS", "projper_ModificationTS"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= E2G (experiment_to_gene) ==================
if _enabled("experiment_to_gene"):
    router.include_router(
        generate_crud_router(
            model=E2G,
            crud_class=E2GCRUD,
            prefix="/experiment_to_gene",
            tag="E2G",
            exclude_fields={"id"},
            create_exclude_fields={"E2G_CreationTS", "E2G_ModificationTS"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= Job ==============================
if _enabled("jobs"):
    router.include_router(
        generate_crud_router(
            model=Job,
            crud_class=JobCRUD,
            prefix="/jobs",
            tag="Job",
            exclude_fields={"id"},
            create_exclude_fields={"job_CreationTS", "job_ModificationTS", "started_at", "finished_at"},
            route_prefix_by_table=_ROUTE_PREFIX_MAP,
        )
    )


# ========================= Convenience: E2G by relations ==========
from sqlalchemy import func as _sa_func


if _enabled("experiment_to_gene"):

    @router.post("/experiment_to_gene/bulk")
    def bulk_e2g(
        payload: list[dict],
        upsert: bool = Query(default=True),
        db: Session = Depends(get_session_dep),
    ):
        """Bulk insert or upsert E2G rows.

        Accepts a list of E2G-like dicts. When ``upsert`` is True, matches on
        (experiment_run_id, gene, geneidtype, label) and updates selected fields;
        otherwise inserts new rows.
        """
        crud = E2GCRUD()
        rows = payload
        if upsert:
            result = crud.bulk_upsert(db, rows)
        else:
            objs = crud.bulk_create(db, rows)
            result = {"inserted": len(objs), "updated": 0}
        return result


if _enabled("experiment_to_gene") and _enabled("experiment_runs"):

    @router.post("/experiment_runs/{run_id}/genes/bulk")
    def bulk_e2g_for_run(
        run_id: int,
        payload: list[dict],
        upsert: bool = Query(default=True),
        db: Session = Depends(get_session_dep),
    ):
        crud = E2GCRUD()
        rows = [dict(r, experiment_run_id=run_id) for r in payload]
        if upsert:
            result = crud.bulk_upsert(db, rows)
        else:
            objs = crud.bulk_create(db, rows)
            result = {"inserted": len(objs), "updated": 0}
        return result


if _enabled("experiments"):

    @router.get("/experiments/by_project/{project_id}")
    def list_experiments_by_project(
        project_id: int,
        limit: int = Query(default=200, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        rows = (
            db.query(Experiment)
            .filter(Experiment.project_id == project_id)
            .order_by(Experiment.id.asc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        Read = make_pydantic_model_from_sqlalchemy(Experiment, name_suffix="Read")
        return [Read.model_validate(r).model_dump() for r in rows]


if _enabled("project_comment"):

    @router.get("/project_comment/by_project/{project_id}")
    def list_comments_by_project(
        project_id: int,
        limit: int = Query(default=200, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        from ispec.db.models import Person as _Person

        rows = (
            db.query(ProjectComment, _Person)
            .outerjoin(_Person, ProjectComment.person_id == _Person.id)
            .filter(ProjectComment.project_id == project_id)
            .order_by(ProjectComment.id.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        Read = make_pydantic_model_from_sqlalchemy(ProjectComment, name_suffix="Read")
        payload = []
        for comment, person in rows:
            item = Read.model_validate(comment).model_dump()
            if person is not None:
                first = getattr(person, "ppl_Name_First", "") or ""
                last = getattr(person, "ppl_Name_Last", "") or ""
                label = f"{last}, {first}".strip().strip(",")
                item["person_label"] = label or str(person.id)
            else:
                item["person_label"] = None
            payload.append(item)
        return payload


if _enabled("project_person"):

    @router.get("/project_person/by_project/{project_id}")
    def list_project_people_by_project(
        project_id: int,
        limit: int = Query(default=200, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        from ispec.db.models import Person as _Person

        rows = (
            db.query(ProjectPerson, _Person)
            .outerjoin(_Person, ProjectPerson.person_id == _Person.id)
            .filter(ProjectPerson.project_id == project_id)
            .order_by(ProjectPerson.id.asc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        Read = make_pydantic_model_from_sqlalchemy(ProjectPerson, name_suffix="Read")
        payload = []
        for project_person, person in rows:
            item = Read.model_validate(project_person).model_dump()
            if person is not None:
                first = getattr(person, "ppl_Name_First", "") or ""
                last = getattr(person, "ppl_Name_Last", "") or ""
                label = f"{last}, {first}".strip().strip(",")
                item["person_label"] = label or str(person.id)
            else:
                item["person_label"] = None
            payload.append(item)
        return payload


if _enabled("experiment_runs"):

    @router.get("/experiment_runs/by_experiment/{experiment_id}")
    def list_runs_by_experiment(
        experiment_id: int,
        limit: int = Query(default=200, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        rows = (
            db.query(ExperimentRun)
            .filter(ExperimentRun.experiment_id == experiment_id)
            .order_by(ExperimentRun.id.asc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        Read = make_pydantic_model_from_sqlalchemy(ExperimentRun, name_suffix="Read")
        return [Read.model_validate(r).model_dump() for r in rows]


if _enabled("experiment_runs") and _enabled("experiments"):

    @router.get("/experiment_runs/by_project/{project_id}")
    def list_runs_by_project(
        project_id: int,
        limit: int = Query(default=500, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        query = (
            db.query(ExperimentRun)
            .join(Experiment, ExperimentRun.experiment_id == Experiment.id)
            .filter(Experiment.project_id == project_id)
            .order_by(ExperimentRun.id.asc())
        )
        rows = query.limit(limit).offset(offset).all()
        Read = make_pydantic_model_from_sqlalchemy(ExperimentRun, name_suffix="Read")
        return [Read.model_validate(r).model_dump() for r in rows]


if _enabled("experiment_to_gene"):

    @router.get(
        "/experiment_to_gene/by_experiment/{experiment_id}", response_model=list[E2GRead]
    )
    def list_genes_by_experiment(
        experiment_id: int,
        q: str | None = None,
        geneidtype: str | None = None,
        limit: int = Query(default=200, ge=1, le=2000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        from ispec.db.models import ExperimentRun as _Run

        query = db.query(E2G).join(_Run, E2G.experiment_run_id == _Run.id).filter(
            _Run.experiment_id == experiment_id
        )
        if q:
            query = query.filter(E2G.gene.ilike(f"%{q}%"))
        if geneidtype:
            query = query.filter(E2G.geneidtype == geneidtype)
        rows = query.order_by(E2G.id.asc()).limit(limit).offset(offset).all()
        return [E2GRead.model_validate(r).model_dump() for r in rows]


    @router.get(
        "/experiment_to_gene/by_project/{project_id}", response_model=list[E2GRead]
    )
    def list_genes_by_project(
        project_id: int,
        q: str | None = None,
        geneidtype: str | None = None,
        limit: int = Query(default=500, ge=1, le=5000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        from ispec.db.models import ExperimentRun as _Run, Experiment as _Exp

        query = (
            db.query(E2G)
            .join(_Run, E2G.experiment_run_id == _Run.id)
            .join(_Exp, _Run.experiment_id == _Exp.id)
            .filter(_Exp.project_id == project_id)
        )
        if q:
            query = query.filter(E2G.gene.ilike(f"%{q}%"))
        if geneidtype:
            query = query.filter(E2G.geneidtype == geneidtype)
        rows = query.order_by(E2G.id.asc()).limit(limit).offset(offset).all()
        return [E2GRead.model_validate(r).model_dump() for r in rows]


if _enabled("jobs"):

    # ========================= Job convenience + transitions ===============
    @router.get("/jobs/by_run/{run_id}")
    def list_jobs_by_run(
        run_id: int,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_session_dep),
    ):
        from ispec.db.models import Job as _Job
        rows = (
            db.query(_Job)
            .filter(_Job.experiment_run_id == run_id)
            .order_by(_Job.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        Read = make_pydantic_model_from_sqlalchemy(Job, name_suffix="Read")
        return [Read.model_validate(r).model_dump() for r in rows]


    @router.post("/jobs/{job_id}/start")
    def start_job(job_id: int, db: Session = Depends(get_session_dep)):
        crud = JobCRUD()
        obj = crud.start(db, job_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Job not found")
        Read = make_pydantic_model_from_sqlalchemy(Job, name_suffix="Read")
        return Read.model_validate(obj).model_dump()


    @router.post("/jobs/{job_id}/succeed")
    def succeed_job(job_id: int, message: str | None = None, db: Session = Depends(get_session_dep)):
        crud = JobCRUD()
        obj = crud.succeed(db, job_id, message)
        if not obj:
            raise HTTPException(status_code=404, detail="Job not found")
        Read = make_pydantic_model_from_sqlalchemy(Job, name_suffix="Read")
        return Read.model_validate(obj).model_dump()


    @router.post("/jobs/{job_id}/fail")
    def fail_job(job_id: int, message: str | None = None, db: Session = Depends(get_session_dep)):
        crud = JobCRUD()
        obj = crud.fail(db, job_id, message)
        if not obj:
            raise HTTPException(status_code=404, detail="Job not found")
        Read = make_pydantic_model_from_sqlalchemy(Job, name_suffix="Read")
        return Read.model_validate(obj).model_dump()

# Convenience read models for list endpoints
