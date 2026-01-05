"""CRUD helper classes and utilities for working with SQLAlchemy models."""

# crud.py
import json
from typing import Any, Iterable, List, Optional, Sequence

from sqlalchemy import select, func, cast, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.sql import sqltypes as T  # canonical type classes (String, Text, etc.)

from ispec.logging import get_logger

# Optional gene identifier normalizer (file-driven; may be None)
try:
    from ispec.genomics.identifiers import GeneNormalizer, TackleGeneNormalizer  # type: ignore
except Exception:  # pragma: no cover - optional component
    GeneNormalizer = None  # type: ignore
    TackleGeneNormalizer = None  # type: ignore

_GENE_NORMALIZER = None


def _get_gene_normalizer():  # pragma: no cover - trivial
    global _GENE_NORMALIZER
    if _GENE_NORMALIZER is None:
        # 1) file-driven mapping via env
        if GeneNormalizer is not None:
            try:
                _GENE_NORMALIZER = GeneNormalizer.from_env()
            except Exception:
                _GENE_NORMALIZER = None
        # 2) tackle-based mapping as a fallback
        if _GENE_NORMALIZER is None and TackleGeneNormalizer is not None:
            try:
                if TackleGeneNormalizer.available():
                    _GENE_NORMALIZER = TackleGeneNormalizer()
            except Exception:
                _GENE_NORMALIZER = None
    return _GENE_NORMALIZER


from ispec.db.models import (
    Person,
    Project,
    ProjectPerson,
    LetterOfSupport,
    ProjectComment,
    E2G,
    ExperimentRun,
    Experiment,
    Job,
    JobStatus,
    PSM,
    MSRawFile,
)


logger = get_logger(__file__)


class CRUDBase:
    """Base class providing common CRUD helpers for SQLAlchemy models."""

    prefix = None

    def __init__(self, model, req_cols: Optional[List[str]] = None):
        self.model = model
        self.req_cols = req_cols

    def get_columns(self):
        """Return the list of column names defined on the mapped table."""
        return [col.name for col in self.model.__table__.columns]

    def clean_input(self, record: dict | None) -> dict:
        # Only keep known columns
        allowed_keys = self.get_columns()
        prefix = self.prefix or ""
        cleaned_record = {}
        for k, v in (record or {}).items():
            if k in allowed_keys:
                cleaned_record[k] = v
                continue
            candidate = f"{prefix}{k}"
            if prefix and candidate in allowed_keys:
                cleaned_record[candidate] = v
                continue
            logger.warning(f"Key '{k}' not in model columns, removing from record.")

        # cleaned_record = {k: v for k, v in record.items() if k in allowed_keys}
        logger.debug(f"Cleaned record keys: {cleaned_record.keys()}")
        return cleaned_record

    def validate_input(self, session: Session, record: dict = None) -> dict:
        """Validate and normalize user supplied data before persistence.

        The method removes keys that do not correspond to model columns, applies
        the optional :attr:`prefix` mapping for alternate names, enforces any
        required columns specified via ``req_cols``, and logs unexpected keys.
        ``record`` must be a dictionary, and ``session`` must be an active
        SQLAlchemy session to support downstream lookups in subclasses.
        """

        if session is None:
            raise ValueError("A database session is required for validation.")

        cleaned_record = self.clean_input(record)

        # Check required columns
        if self.req_cols is not None:
            for col in self.req_cols:
                if col not in cleaned_record:
                    raise ValueError(f"{col} not in input record")

        return cleaned_record

    def get(self, session: Session, id: int):
        """Return the instance with the given primary key or ``None``."""
        return session.query(self.model).filter(self.model.id == id).first()

    def create(self, session: Session, record: dict):
        """Insert a new row into the database and return the resulting object."""
        record = self.validate_input(session, record)
        if record is None:
            logger.warning("Record is None after validation, skipping insert.")
            return None
        obj = self.model(**record)
        session.add(obj)
        session.commit()
        session.refresh(obj)
        logger.info(f"Inserted into {self.model.__tablename__}: {record}")
        return obj

    def bulk_create(self, session: Session, records: List[dict]):
        """Insert multiple records in one transaction and return the objects."""
        if not records:
            return []
        cleaned = [self.validate_input(session, r) for r in records]
        cleaned = list(filter(None, cleaned))  # Remove None values
        objs = [self.model(**r) for r in cleaned]
        session.add_all(objs)
        session.commit()
        return objs

    def delete(self, session: Session, id: int) -> bool:
        """Delete the row with the supplied id, returning ``True`` on success."""
        obj = self.get(session, id)
        if obj:
            session.delete(obj)
            session.commit()
            return True
        return False

    def readonly_fields(self) -> set[str]:
        columns = self.get_columns()
        readonly_suffixes = ("_CreationTS", "_ModificationTS", "_LegacyImportTS")
        readonly: set[str] = {"id"}
        for name in columns:
            lowered = name.lower()
            if any(lowered.endswith(suffix.lower()) for suffix in readonly_suffixes):
                readonly.add(name)
            if lowered.endswith("displayid") or lowered.endswith("displaytitle"):
                readonly.add(name)
            if "foundcount" in lowered:
                readonly.add(name)
        return readonly

    def after_update(self, session: Session, obj: Any, updates: dict[str, Any]) -> None:
        return None

    def update(self, session: Session, obj: Any, record: dict) -> Any:
        updates = self.clean_input(record)
        readonly = self.readonly_fields()
        for field, value in updates.items():
            if field in readonly:
                continue
            setattr(obj, field, value)

        self.after_update(session, obj, updates)

        session.commit()
        session.refresh(obj)
        return obj



    # hook: expression used for label; override per model if needed
    def label_expr(self):
        """Default human label for options(). Override per model if needed."""
        M = self.model
        cols = M.__table__.columns.keys()

        # helpers: COALESCE to avoid NULL-propagation in concatenation
        def coalesce(colname: str, fallback: str = ""):
            return func.coalesce(getattr(M, colname), fallback)

        if 'name' in cols:
            return getattr(M, 'name')
        if 'title' in cols:
            return getattr(M, 'title')
        if {'first_name', 'last_name'}.issubset(cols):
            # "Last, First" and trim extra spaces if one side empty
            return func.trim(coalesce('last_name') + ', ' + coalesce('first_name'))
        if 'code' in cols:
            return getattr(M, 'code')

        # fallback: cast id to text
        return cast(getattr(M, 'id'), T.String())



    def list_options(
        self,
        db: Session,
        *,
        q: str | None = None,
        limit: int = 20,
        ids: Sequence[int] | None = None,
        exclude_ids: Sequence[int] | None = None,
        order: str | None = None
    ) -> list[dict[str, Any]]:
        M = self.model
        lbl = self.label_expr().label('label')

        id_col = getattr(M, 'id').label('value')
        stmt = select(id_col, lbl)

        if ids:
            stmt = stmt.where(getattr(M, 'id').in_(ids))
        if exclude_ids:
            stmt = stmt.where(~getattr(M, 'id').in_(exclude_ids))

        if q:
            # Case-insensitive match; .ilike turns into LIKE on SQLite (still case-insensitive)
            stmt = stmt.where(lbl.ilike(f"%{q}%"))

        # Sort by label unless caller wants something else
        if order == 'id':
            stmt = stmt.order_by(getattr(M, 'id').asc())
        else:
            stmt = stmt.order_by(lbl.asc())

        stmt = stmt.limit(limit)
        rows: Iterable[tuple[int, str]] = db.execute(stmt).all()
        return [{'value': v, 'label': l} for (v, l) in rows]



class PersonCRUD(CRUDBase):

    # prefix = "ppl_"

    def __init__(self):
        super().__init__(Person, req_cols=["ppl_Name_Last", "ppl_Name_First"])

    def validate_input(self, session: Session, record: dict) -> dict | None:

        if session is None:
            raise ValueError("A database session is required for validation.")

        record = super().validate_input(session, record)

        last_name = record.get("ppl_Name_Last", "").strip().lower()
        first_name = record.get("ppl_Name_First", "").strip().lower()

        if not last_name:
            return None  # or raise ValueError if you want it to fail hard

        # Check for case-insensitive match using SQL functions
        existing = (
            session.query(self.model)
            .filter(
                func.lower(self.model.ppl_Name_Last) == last_name,
                func.lower(self.model.ppl_Name_First) == first_name,
            )
            .first()
        )

        if existing:
            logger.info(
                f"Person with name '{first_name} {last_name}' already exists (ID {existing.id}). Skipping insert."
            )
            return None

        return record

    def create(self, session: Session, record: dict):
        validated = self.validate_input(session, record)
        if validated is None:
            return None  # or return existing object or ID if desired
        return super().create(session, validated)



    def label_expr(self): # override base label_expr 
        # e.g., "Lastname, Firstname (email)"
        cols = self.model.__table__.columns.keys()
        expr = None
        if {'ppl_Name_Last','ppl_Name_First'}.issubset(cols):
            expr = getattr(self.model, 'ppl_Name_Last') + ', ' + getattr(self.model, 'ppl_Name_First')
        else:
            expr = super().label_expr()
        if 'ppl_email' in cols:  #this might be the wrong form name
            expr = expr + ' (' + getattr(self.model, 'ppl_email') + ')'
        return func.trim(expr)

    # def label_expr(self):
    # is this one safe ? above attr check maybe better?
    #     M = self.model
    #     return func.trim(
    #         func.coalesce(M.last_name, '') + ', ' +
    #         func.coalesce(M.first_name, '')
    #     )


class ProjectCRUD(CRUDBase):

    # prefix = "prj_"

    def __init__(self):
        super().__init__(
            Project, req_cols=["prj_ProjectTitle"]
        )

    @staticmethod
    def _ensure_display_fields(project: Project) -> None:
        display_id = getattr(project, "prj_PRJ_DisplayID", None)
        if not (isinstance(display_id, str) and display_id.strip()):
            project.prj_PRJ_DisplayID = f"MSPC{project.id:06d}"

        display_title = getattr(project, "prj_PRJ_DisplayTitle", None)
        if not (isinstance(display_title, str) and display_title.strip()):
            title = getattr(project, "prj_ProjectTitle", "") or ""
            project.prj_PRJ_DisplayTitle = f"{project.prj_PRJ_DisplayID} - {title}".strip()

    def label_expr(self):
        cols = self.model.__table__.columns.keys()
        if "prj_ProjectTitle" in cols:
            return getattr(self.model, "prj_ProjectTitle")
        return super().label_expr()

    def validate_input(self, session: Session, record: dict) -> dict | None:
        record = super().validate_input(session, record)

        title = record.get("prj_ProjectTitle", "").strip().lower()
        background = record.get("prj_ProjectBackground", "").strip().lower()

        if not title:
            return None

        # Check for existing project by lowercased title
        existing = (
            session.query(self.model)
            .filter(func.lower(self.model.prj_ProjectTitle) == title)
            .first()
        )

        if existing:
            logger.info(
                f"Project with title '{title}' already exists (ID {existing.id}). Skipping insert."
            )
            return None

        return record

    def create(self, session: Session, record: dict):
        validated = self.validate_input(session, record)
        if validated is None:
            return None

        obj = self.model(**validated)
        session.add(obj)
        session.flush()
        self._ensure_display_fields(obj)
        session.commit()
        session.refresh(obj)
        logger.info(f"Inserted into {self.model.__tablename__}: {validated}")
        return obj

    def after_update(self, session: Session, project: Project, updates: dict[str, Any]) -> None:
        self._ensure_display_fields(project)
        if "prj_ProjectTitle" in updates:
            title = getattr(project, "prj_ProjectTitle", "") or ""
            project.prj_PRJ_DisplayTitle = f"{project.prj_PRJ_DisplayID} - {title}".strip()


class ProjectCommentCRUD(CRUDBase):

    # prefix = "com_"
    TABLE = "project_comment"  # might not be using this anymore
    REQ_COLS = (
        "project_id",
        "person_id",
    )

    def __init__(self):
        super().__init__(ProjectComment, req_cols=self.REQ_COLS)

    def validate_input(self, session, record: dict) -> dict:
        record = super().validate_input(session, record)
        prj_id = record.get("project_id")
        ppl_id = record.get("person_id")

        if prj_id is None:
            raise ValueError("project_id is required for ProjectComment")
        if prj_id is not None:
            # Check if project exists
            project_exists = session.query(Project).filter_by(id=prj_id).first()
            if not project_exists:
                raise ValueError(f"Invalid project_id: {prj_id}")
        if ppl_id is not None:
            # Check if person exists
            person_exists = session.query(Person).filter_by(id=ppl_id).first()
            if not person_exists:
                raise ValueError(f"Invalid person_id: {ppl_id}")
        return record


class ProjectPersonCRUD(CRUDBase):
    def __init__(self):
        return super().__init__(ProjectPerson)

    def validate_input(self, session: Session, record: dict = None) -> dict:
        person_id = record.get("person_id")
        project_id = record.get("project_id")

        if session is None:
            raise ValueError("A database session is required for validation.")

        if not session.query(Person).filter_by(id=person_id).first():
            raise ValueError(f"Invalid person_id: {person_id}")

        if not session.query(Project).filter_by(id=project_id).first():
            raise ValueError(f"Invalid project_id: {project_id}")

        return super().validate_input(session, record)

    def create(self, session: Session, data: dict) -> ProjectPerson:
        validated = self.validate_input(session, data)
        if validated is None:
            logger.warning("Record is None after validation, skipping insert.")
            return None
        return super().create(session, validated)


class LetterOfSupportCRUD(CRUDBase):
    def __init__(self):
        return super().__init__(LetterOfSupport)


class E2GCRUD(CRUDBase):
    def __init__(self):
        super().__init__(
            E2G,
            req_cols=[
                "experiment_run_id",
                "gene",
                "geneidtype",
            ],
        )
        self._normalizer = None  # lazy-load to avoid heavy import costs
        self._validated_run_ids: set[int] = set()

    def _equivalent_pairs(self, gene: str, geneidtype: str) -> list[tuple[str, str]]:
        pairs = [(geneidtype, gene)]
        norm = self._normalizer or _get_gene_normalizer()
        self._normalizer = norm
        if norm is None:
            return pairs
        try:
            equivalents = norm.equivalents(gene, geneidtype) or []
            combined = pairs + list(equivalents)
            seen: set[tuple[str, str]] = set()
            out: list[tuple[str, str]] = []
            for pair in combined:
                try:
                    key = (str(pair[0]), str(pair[1]))
                except Exception:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
            return out or pairs
        except Exception:
            return pairs

    def validate_input(
        self,
        session: Session,
        record: dict | None,
        *,
        allow_existing: bool = False,
    ) -> dict | None:
        if session is None:
            raise ValueError("A database session is required for validation.")
        if record is None:
            return None

        cleaned = super().validate_input(session, record)
        exp_run_id = cleaned.get("experiment_run_id")
        if exp_run_id is None:
            raise ValueError("experiment_run_id is required for E2G")
        if exp_run_id not in self._validated_run_ids:
            exists = session.query(ExperimentRun).filter_by(id=exp_run_id).first()
            if not exists:
                raise ValueError(f"Invalid experiment_run_id: {exp_run_id}")
            self._validated_run_ids.add(int(exp_run_id))

        gene = (cleaned.get("gene") or "").strip()
        geneidtype = (cleaned.get("geneidtype") or "").strip()
        if not gene or not geneidtype:
            return None

        label = cleaned.get("label")
        if label is None:
            label = "0"
        cleaned["label"] = str(label).strip() or "0"
        cleaned["gene"] = gene
        cleaned["geneidtype"] = geneidtype

        if allow_existing:
            return cleaned

        # Best-effort duplicate check (per run, identifier, label).
        pairs = self._equivalent_pairs(gene, geneidtype)
        conds = [and_(E2G.geneidtype == t, E2G.gene == v) for (t, v) in pairs]
        dup = (
            session.query(E2G)
            .filter(E2G.experiment_run_id == exp_run_id, E2G.label == cleaned["label"], or_(*conds))
            .first()
        )
        if dup:
            return None

        return cleaned

    def label_expr(self):
        cols = self.model.__table__.columns.keys()
        if "gene_symbol" in cols:
            return func.coalesce(getattr(self.model, "gene_symbol"), getattr(self.model, "gene"))
        return getattr(self.model, "gene")

    def bulk_upsert(
        self,
        session: Session,
        records: list[dict],
        *,
        update_fields: tuple[str, ...] = (
            "gene_symbol",
            "description",
            "taxon_id",
            "sra",
            "psms",
            "psms_u2g",
            "peptide_count",
            "peptide_count_u2g",
            "coverage",
            "coverage_u2g",
            "area_sum_u2g_0",
            "area_sum_u2g_all",
            "area_sum_max",
            "area_sum_dstrAdj",
            "iBAQ_dstrAdj",
            "peptideprint",
            "metadata_json",
        ),
    ) -> dict[str, int]:
        inserted = 0
        updated = 0

        def merge_metadata_json(existing_json: str | None, new_json: str) -> str:
            try:
                left = json.loads(existing_json) if existing_json else {}
            except Exception:
                left = {}
            try:
                right = json.loads(new_json)
            except Exception:
                return new_json
            if not isinstance(left, dict) or not isinstance(right, dict):
                return new_json
            merged = dict(left)
            merged.update(right)
            return json.dumps(merged, ensure_ascii=False, separators=(",", ":"))

        for rec in records:
            rec = dict(rec)
            rec.setdefault("label", "0")
            validated = self.validate_input(session, rec, allow_existing=True)
            if validated is None:
                continue

            exp_run_id = validated.get("experiment_run_id")
            gene = (validated.get("gene") or "").strip()
            geneidtype = (validated.get("geneidtype") or "").strip()
            label = (validated.get("label") or "0").strip()

            pairs = self._equivalent_pairs(gene, geneidtype)
            conds = [and_(E2G.geneidtype == t, E2G.gene == v) for (t, v) in pairs]
            existing = (
                session.query(E2G)
                .filter(E2G.experiment_run_id == exp_run_id, E2G.label == label, or_(*conds))
                .first()
            )
            if existing:
                changed = False
                for field in update_fields:
                    if field not in validated:
                        continue
                    if not hasattr(existing, field):
                        continue
                    value = validated[field]
                    if field == "metadata_json" and value is not None:
                        value = merge_metadata_json(getattr(existing, field, None), str(value))
                    if getattr(existing, field) != value:
                        setattr(existing, field, value)
                        changed = True
                if changed:
                    session.add(existing)
                    updated += 1
            else:
                session.add(self.model(**validated))
                inserted += 1

        session.commit()
        return {"inserted": inserted, "updated": updated}


class PSMCRUD(CRUDBase):
    def __init__(self):
        super().__init__(PSM)

    def label_expr(self):
        M = self.model
        cols = M.__table__.columns.keys()
        parts = []
        if "peptide" in cols:
            parts.append(getattr(M, "peptide"))
        if "charge" in cols:
            parts.append(func.coalesce(func.cast(getattr(M, "charge"), T.String()), ""))
        if parts:
            # peptide(+charge) for readability
            return func.trim(parts[0] + " +" + parts[1])
        return super().label_expr()


class MSRawFileCRUD(CRUDBase):
    def __init__(self):
        super().__init__(MSRawFile)

    def label_expr(self):
        cols = self.model.__table__.columns.keys()
        if "uri" in cols:
            return getattr(self.model, "uri")
        return super().label_expr()


class ExperimentCRUD(CRUDBase):
    def __init__(self):
        super().__init__(Experiment)

    def readonly_fields(self) -> set[str]:
        readonly = super().readonly_fields()
        readonly.update({"project_id", "record_no", "exp_Data_FLAG", "exp_exp2gene_FLAG"})
        return readonly

    def label_expr(self):
        cols = self.model.__table__.columns.keys()
        if "record_no" in cols:
            return getattr(self.model, "record_no")
        return super().label_expr()

    def validate_input(self, session: Session, record: dict | None) -> dict | None:
        if session is None:
            raise ValueError("A database session is required for validation.")
        if record is None:
            return None

        project_id = record.get("project_id")
        if project_id is None:
            raise ValueError("project_id is required for Experiment")
        exists = session.query(Project).filter_by(id=project_id).first()
        if not exists:
            raise ValueError(f"Invalid project_id: {project_id}")

        # basic duplicate check on record_no within a project
        rec = (record.get("record_no") or "").strip()
        if rec:
            dup = (
                session.query(Experiment)
                .filter_by(project_id=project_id, record_no=rec)
                .first()
            )
            if dup:
                return None

        return super().validate_input(session, record)


class ExperimentRunCRUD(CRUDBase):
    def __init__(self):
        super().__init__(ExperimentRun)

    def readonly_fields(self) -> set[str]:
        readonly = super().readonly_fields()
        readonly.update({"experiment_id", "run_no", "search_no"})
        return readonly

    def label_expr(self):
        M = self.model
        cols = M.__table__.columns.keys()
        if {"experiment_id", "run_no", "search_no", "label"}.issubset(cols):
            return func.trim(
                cast(getattr(M, "experiment_id"), T.String())
                + "-"
                + cast(getattr(M, "run_no"), T.String())
                + "-"
                + cast(getattr(M, "search_no"), T.String())
                + "-"
                + func.coalesce(cast(getattr(M, "label"), T.String()), "")
            )
        return super().label_expr()

    def validate_input(self, session: Session, record: dict | None) -> dict | None:
        if session is None:
            raise ValueError("A database session is required for validation.")
        if record is None:
            return None

        experiment_id = record.get("experiment_id")
        if experiment_id is None:
            raise ValueError("experiment_id is required for ExperimentRun")
        exists = session.query(Experiment).filter_by(id=experiment_id).first()
        if not exists:
            raise ValueError(f"Invalid experiment_id: {experiment_id}")

        # avoid unique constraint violation by pre-checking
        run_no = record.get("run_no", 1)
        search_no = record.get("search_no", 1)
        label = record.get("label")
        if label is None:
            label = "0"
        if isinstance(label, str):
            label = label.strip() or "0"
        else:
            label = str(label)
        record["label"] = label
        dup = (
            session.query(ExperimentRun)
            .filter_by(
                experiment_id=experiment_id, run_no=run_no, search_no=search_no, label=label
            )
            .first()
        )
        if dup:
            return None

        return super().validate_input(session, record)


class JobCRUD(CRUDBase):
    def __init__(self):
        super().__init__(Job)

    def validate_input(self, session: Session, record: dict | None) -> dict | None:
        if session is None:
            raise ValueError("A database session is required for validation.")
        if record is None:
            return None
        run_id = record.get("experiment_run_id")
        if run_id is None:
            raise ValueError("experiment_run_id is required for Job")
        exists = session.query(ExperimentRun).filter_by(id=run_id).first()
        if not exists:
            raise ValueError(f"Invalid experiment_run_id: {run_id}")
        return super().validate_input(session, record)

    def start(self, session: Session, job_id: int) -> Job | None:
        obj = self.get(session, job_id)
        if not obj:
            return None
        obj.status = JobStatus.running
        from datetime import datetime, UTC
        obj.started_at = datetime.now(UTC)
        session.commit()
        session.refresh(obj)
        return obj

    def succeed(self, session: Session, job_id: int, message: str | None = None) -> Job | None:
        obj = self.get(session, job_id)
        if not obj:
            return None
        obj.status = JobStatus.succeeded
        if message:
            obj.message = message
        from datetime import datetime, UTC
        obj.finished_at = datetime.now(UTC)
        session.commit()
        session.refresh(obj)
        return obj

    def fail(self, session: Session, job_id: int, message: str | None = None) -> Job | None:
        obj = self.get(session, job_id)
        if not obj:
            return None
        obj.status = JobStatus.failed
        if message:
            obj.message = message
        from datetime import datetime, UTC
        obj.finished_at = datetime.now(UTC)
        session.commit()
        session.refresh(obj)
        return obj


# class ProjectComment(TableCRUD):
#     TABLE = "project_comment"
#     REQ_COLS = ("i_id",)

#     def validate_input(self, record: dict):
#         record = super().validate_input(record)
#         i_id = record.get("i_id")
#         if not id:
#             return None
#         return record


#
# ==
# class TableCRUD:
#
#     TABLE = None
#     REQ_COLS = None
#
#     def __init__(self, conn: sqlite3.Connection = None, table_name: str = None):
#         self.table = table_name or self.TABLE
#         self.conn = conn
#
#     @cache
#     def get_columns(self):
#         result = self.conn.execute(f"PRAGMA table_info({self.table})")
#         return [row[1] for row in result.fetchall()]
#
#     def validate_input(self, record: dict):
#         if hasattr(self.REQ_COLS, "__iter__"):
#             for col in self.REQ_COLS:
#                 if col not in record.keys():
#                     raise ValueError(f"{col} not in input records")
#         cleaned_record = {k: v for k, v in record.items() if k in self.get_columns()}
#         logger.debug(f"cleaned record keys {cleaned_record.keys()}")
#         return cleaned_record
#
#     def insert(self, record: Dict[str, str]) -> int:
#
#         self.validate_input(record)
#
#         keys = list(record.keys())
#         values = [record[k] for k in keys]
#         placeholders = ", ".join(["?"] * len(keys))
#         columns = ", ".join(keys)
#
#         self.conn.execute(
#             f"INSERT INTO {self.table} ({columns}) VALUES ({placeholders})", values
#         )
#         self.conn.commit()
#         logger.info(f"Inserted into {self.table}: {record}")
#         return self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
#
#     def bulk_insert(self, records: list[dict]) -> None:
#         # needs testing
#         if not records:
#             return
#         cleaned_records = list(
#             filter(None, [self.validate_input(record) for record in records])
#         )
#         if not cleaned_records:
#             return
#         # import pdb; pdb.set_trace()
#         # self.validate_input(records[0])  # assume all rows follow same structure
#
#         keys = list(cleaned_records[0].keys())
#         placeholders = ", ".join(["?"] * len(keys))
#         columns = ", ".join(keys)
#
#         values = [[r[k] for k in keys] for r in cleaned_records]
#         self.conn.executemany(
#             f"INSERT INTO {self.table} ({columns}) VALUES ({placeholders})", values
#         )
#         self.conn.commit()
#
#     def get_by_id(self, record_id: int) -> Optional[Dict[str, str]]:
#         result = self.conn.execute(
#             f"SELECT * FROM {self.table} WHERE id = ?", (record_id,)
#         ).fetchone()
#         return dict(result) if result else None
#
#     def delete_by_id(self, record_id: int) -> bool:
#         result = self.conn.execute(
#             f"DELETE FROM {self.table} WHERE id = ?", (record_id,)
#         )
#         self.conn.commit()
#         return result.rowcount > 0
#


class TableCRUD:
    pass


# class Person(TableCRUD):

#     TABLE = "person"
#     REQ_COLS = ("ppl_Name_Last", "ppl_Name_First")

#     def validate_input(self, record: dict):
#         record = super().validate_input(record)
#         last_name = record.get("ppl_Name_Last", "")
#         first_name = record.get("ppl_Name_First", "")
#         if not last_name:
#             return None
#         last_name = last_name.strip().lower()
#         first_name = first_name.strip().lower()

#         # Check for existing person (case-insensitive)
#         result = self.conn.execute(
#             "SELECT id FROM person WHERE LOWER(ppl_Name_Last) = ? AND LOWER(ppl_Name_First) = ? LIMIT 1",
#             (last_name, first_name),
#         ).fetchone()
#         # import pdb; pdb.set_trace()

#         if result:
#             logger.info(
#                 f"Person with last name '{last_name}' already exists (ID {result[0]}). Skipping insert."
#             )
#             return None
#             # return result[0]  # existing ID

#         return record


# class Project(TableCRUD):
#
#     TABLE = "project"
#     REQ_COLS = (
#         "prj_ProjectTitle",
#         "prj_ProjectBackground",
#     )
#
#     def validate_input(self, record: dict):
#         # rewrite for project specifi
#         # check if project title matches an existing project title in the database
#
#         record = super().validate_input(record)
#
#         proj_ProjectTitle = record.get("prj_ProjectTitle", "")
#         proj_ProjectBackground = record.get("proj_ProjectBackground", "")
#         if not proj_ProjectTitle:
#             return None
#         proj_ProjectTitle = proj_ProjectTitle.strip().lower()
#         proj_ProjectBackground = proj_ProjectBackground.strip().lower()
#         ##
#
#         # Check for existing person (case-insensitive)
#         result = self.conn.execute(
#             "SELECT id FROM project WHERE prj_ProjectTitle = ? LIMIT 1",
#             (proj_ProjectTitle,),
#         ).fetchone()
#
#         if result:
#             logger.info(
#                 f"Project with title '{proj_ProjectTitle}' already exists (ID {result[0]}). Skipping insert."
#             )
#             return None
#             # return result[0]  # existing ID
#
#         return record
#
#
# class ProjectPerson(TableCRUD):
#     TABLE = "project_person"
#     REQ_COLS = ("person_id", "project_id")
#
#     def validate_input(self, record: dict):
#         super().validate_input(record)
#
#         person_id = record.get("person_id")
#         project_id = record.get("project_id")
#
#         person_query = self.conn.execute(
#             "SELECT id FROM person where id = ?", (person_id,)
#         ).fetchone()
#
#         logger.info(f"person query {person_query}")
#         if person_query is None:
#             logger.error(
#                 f"Tried to make a project person link with an invalid person id {person_id}"
#             )
#             raise ValueError(f"{person_id} not present in person table")
#
#         project_query = self.conn.execute(
#             "SELECT id FROM project where id = ?", (project_id,)
#         ).fetchone()
#
#         logger.info(f"project query {project_query}")
#         if project_query is None:
#             logger.error(
#                 f"Tried to make a project person link with an invalid project id {project_id}"
#             )
#             raise ValueError(f"{project_id} not present in projecttable")


class ProjectNote(TableCRUD):
    TABLE = "project_note"
    REQ_COLS = ("i_id",)

    def validate_input(self, record: dict):
        record = super().validate_input(record)
        i_id = record.get("i_id")
        if not i_id:
            return None
        return record


# ADDING DATA TO TABLES.
def insert_df_to_table(conn, table_name, df, column_definitions):
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    table_cols = [row[1] for row in cursor]

    # checking for valid columns. uses primary key id to verify identity as a table.
    common_cols = [
        col
        for col in table_cols
        if col in df.columns
        and not (
            column_definitions[col].get("pk")
            and column_definitions[col].get("sqltype", "").startswith("INTEGER")
        )
    ]

    # return statement if it fails to find requested table/columns
    if not common_cols or df.empty:
        logger.warning("No insertable columns found or empty DataFrame.")
        return

    placeholders = ", ".join(["?"] * len(common_cols))
    q = f"INSERT INTO {table_name} ({', '.join(common_cols)}) VALUES ({placeholders})"
    values = df[common_cols].values.tolist()

    logger.info("Inserting into table %s", table_name)
    logger.info("Columns: %s", common_cols)
    logger.debug("Sample types: %s", [type(x) for x in values[0]])
    logger.debug("Sample row: %s", values[0])

    conn.executemany(q, values)


# How do I test this?


# EXECUTE THIS EVERYTIME A NEW PROJECT OR PERSON IS ADDED
# INSERT DIFFERENT TABLE NAME BASED ON WHAT IS BEING ADDED, ex you're adding a person: table_name will be person
# Make sure the table name added


# ADDING PEOPLE - + checking presence
# ADDING PROJECT - + checkin presence
# MATCHING FOREIGN KEYs people_project, updating tables.
