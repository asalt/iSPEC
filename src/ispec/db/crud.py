# crud.py
from typing import Dict, Optional, List, Sequence, Any
from functools import cache
import sqlite3

from sqlalchemy.orm import Session
from sqlalchemy import func

from sqlalchemy import select, func, cast
from sqlalchemy.sql import sqltypes as T   # canonical type classes (String, Text, etc.)
from sqlalchemy.orm import Session

from ispec.db.connect import get_session
from ispec.logging import get_logger


from ispec.db.models import (
    Person,
    Project,
    ProjectPerson,
    LetterOfSupport,
    ProjectComment,
)


logger = get_logger(__file__)


class CRUDBase:

    prefix = None

    def __init__(self, model, req_cols: Optional[List[str]] = None):
        self.model = model
        self.req_cols = req_cols

    def get_columns(self):
        return [col.name for col in self.model.__table__.columns]

    def validate_input(self, session: Session, record: dict = None) -> dict:

        if session is None:
            raise ValueError("A database session is required for validation.")

        # Only keep known columns
        allowed_keys = self.get_columns()
        cleaned_record = {}
        for k, v in record.items():
            if k not in allowed_keys and f"{self.prefix}{k}" in allowed_keys:
                cleaned_record[f"{self.prefix}{k}"] = v
            elif k not in allowed_keys and f"{self.prefix}{k}" not in allowed_keys:
                logger.warning(f"Key '{k}' not in model columns, removing from record.")
            else:
                cleaned_record[k] = v

        # cleaned_record = {k: v for k, v in record.items() if k in allowed_keys}
        logger.debug(f"Cleaned record keys: {cleaned_record.keys()}")

        # Check required columns
        if self.req_cols is not None:
            for col in self.req_cols:
                if col not in cleaned_record:
                    raise ValueError(f"{col} not in input record")

        return cleaned_record

    def get(self, session: Session, id: int):
        return session.query(self.model).filter(self.model.id == id).first()

    def create(self, session: Session, record: dict):
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
        if not records:
            return []
        cleaned = [self.validate_input(session, r) for r in records]
        cleaned = list(filter(None, cleaned))  # Remove None values
        objs = [self.model(**r) for r in cleaned]
        session.add_all(objs)
        session.commit()
        return objs

    def delete(self, session: Session, id: int) -> bool:
        obj = self.get(session, id)
        if obj:
            session.delete(obj)
            session.commit()
            return True
        return False



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
            Project, req_cols=["prj_ProjectTitle", "prj_ProjectBackground"]
        )

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
        return super().create(session, validated)


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
        if not id:
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
