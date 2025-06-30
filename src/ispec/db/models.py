# db/models.py
import sqlite3
from typing import List
from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    LargeBinary,
    ForeignKey,
    create_engine,
    event,
    ForeignKey,
    LargeBinary,
    create_engine,
    event,
    Text,
    Float,
    DateTime,
    Integer,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.engine import Engine


from sqlalchemy.orm import relationship
from datetime import datetime
import pandas as pd

#

from ispec.logging import get_logger

#

logger = get_logger(__file__)


#
def make_timestamp_mixin(prefix: str):
    fields = {
        f"{prefix}_CreationTS": mapped_column(DateTime, default=datetime.utcnow),
        f"{prefix}_ModificationTS": mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        '__annotations__': {
            f"{prefix}_CreationTS": Mapped[datetime],
            f"{prefix}_ModificationTS": Mapped[datetime]
        }
    }
    return type(f"{prefix.capitalize()}TimestampMixin", (object,), fields)


# Timestamp conversion functions
def adapt_timestamp(ts): # This should work for both pd.Timestamp and datetime.datetime.
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def convert_timestamp(s: bytes):
    return pd.Timestamp(s.decode())

TIMESTAMP_MIXINS = {
    "prj": make_timestamp_mixin("prj"),
    "ppl": make_timestamp_mixin("ppl"),
    "com": make_timestamp_mixin("com"),
    "los": make_timestamp_mixin("los")
}

class Base(DeclarativeBase):
    __table_args__ = {"sqlite_autoincrement": True}


class Person(TIMESTAMP_MIXINS['ppl'], Base):
    __tablename__ = "person"

    id: Mapped[int] = mapped_column(primary_key=True)
    ppl_AddedBy: Mapped[str] = mapped_column(Text)
    # ppl_CreationTS: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # ppl_ModificationTS: Mapped[datetime] = mapped_column(
    #     DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    # )

    ppl_Name_First: Mapped[str] = mapped_column(Text)
    ppl_Name_Last: Mapped[str] = mapped_column(Text)
    ppl_Domain: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Email: Mapped[str] = mapped_column(Text)
    ppl_Phone: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_PI: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Institution: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Center: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Department: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Status: Mapped[str | None] = mapped_column(Text, nullable=True)

    ppl_Roles_PI: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Roles_CoreUser: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Roles_CoreStaff: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Roles_Collaborator: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Roles: Mapped[str | None] = mapped_column(Text, nullable=True)

    comments: Mapped[List["ProjectComment"]] = relationship(back_populates="person")
    projects: Mapped[List["ProjectPerson"]] = relationship(back_populates="person")


class Project(TIMESTAMP_MIXINS['prj'], Base):
    __tablename__ = "project"

    id: Mapped[int] = mapped_column(primary_key=True)
    prj_AddedBy: Mapped[str] = mapped_column(Text)
    # prj_CreationTS: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # prj_ModificationTS: Mapped[datetime] = mapped_column(
    #     DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    # )

    prj_PRJ_DisplayID: Mapped[str] = mapped_column(Text)
    prj_PRJ_DisplayTitle: Mapped[str] = mapped_column(Text)
    prj_iLabs_RequestName: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectTitle: Mapped[str] = mapped_column(Text)
    prj_RnD: Mapped[bool] = mapped_column(Boolean, default=False)
    prj_CancerRelevance: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_CPRIT_RFP: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_PI: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Project_LabContact: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Project_LabPersonnel: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_MSPC_Leader: Mapped[str] = mapped_column(Text)
    prj_MSPC_Personnel_Primary: Mapped[str] = mapped_column(Text)
    prj_MSPC_Personnel: Mapped[str] = mapped_column(Text)
    prj_MSPC_Personnel_Analysis: Mapped[str] = mapped_column(Text)
    prj_Services_Type: Mapped[str] = mapped_column(Text)
    prj_Services: Mapped[str] = mapped_column(Text)
    prj_GrantSupport: Mapped[str] = mapped_column(Text)
    prj_ProjectDomain: Mapped[str] = mapped_column(Text)
    prj_ProjectBackground: Mapped[str] = mapped_column(Text)
    prj_ProjectSuggestions2Customer: Mapped[str] = mapped_column(Text)
    prj_ProjectSamples: Mapped[str] = mapped_column(Text)
    prj_ProjectCoreTasks: Mapped[str] = mapped_column(Text)
    prj_ProjectQuestions: Mapped[str] = mapped_column(Text)
    prj_ProjectDescription: Mapped[str] = mapped_column(Text)
    prj_Project_SampleType: Mapped[str] = mapped_column(Text)
    prj_Project_FuturePossibilities: Mapped[str] = mapped_column(Text)
    prj_RequireGelPix: Mapped[str] = mapped_column(Text)
    prj_Experiments_rf: Mapped[str] = mapped_column(Text)
    prj_RelatedProjects_rf: Mapped[str] = mapped_column(Text)
    prj_Comments_Meeting: Mapped[str] = mapped_column(Text)
    prj_Comments_Billing: Mapped[str] = mapped_column(Text)
    prj_Comments_iLAB: Mapped[str] = mapped_column(Text)
    prj_Status: Mapped[str] = mapped_column(Text)

    prj_Date_Submitted: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    prj_Date_LysatePreparation: Mapped[str] = mapped_column(Text)

    prj_Date_MSPreparation: Mapped[str] = mapped_column(Text)
    prj_RequireGelPix_Sent: Mapped[str] = mapped_column(Text)
    prj_RequireGelPix_Confirmed: Mapped[str] = mapped_column(Text)
    prj_Date_Analysis: Mapped[str] = mapped_column(Text)
    prj_Comments_Specific: Mapped[str] = mapped_column(Text)
    prj_Comments_General: Mapped[str] = mapped_column(Text)
    prj_Date_Review: Mapped[str] = mapped_column(Text)
    prj_Date_Closed: Mapped[str] = mapped_column(Text)
    prj_Comments_Review: Mapped[str] = mapped_column(Text)
    prj_IncludeHandouts: Mapped[str] = mapped_column(Text)
    prj_ProjectCostExplanation: Mapped[str] = mapped_column(Text)
    prj_Service_365_Analysis: Mapped[str] = mapped_column(Text)
    prj_Service_APMS_Experiments: Mapped[str] = mapped_column(Text)
    prj_Service_APMS_Analysis: Mapped[str] = mapped_column(Text)
    prj_Service_PerBand_Experiments: Mapped[str] = mapped_column(Text)
    prj_Service_PerBand_Analysis: Mapped[str] = mapped_column(Text)
    prj_Service_PTM_Analysis: Mapped[str] = mapped_column(Text)
    prj_Service_DesignHours: Mapped[str] = mapped_column(Text)
    prj_ProjectPrice: Mapped[str] = mapped_column(Text)
    prj_Quote: Mapped[str] = mapped_column(Text)
    prj_Invoice: Mapped[str] = mapped_column(Text)
    prj_Invoice_Date: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True),
    prj_Billing_Date: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True),
    prj_PaymentReceived: Mapped[bool] = mapped_column(Boolean, default=False)
    prj_ProjectDomainMSPC: Mapped[str] = mapped_column(Text, default=None, nullable=True)
    prj_Project_SampleHandling: Mapped[str] = mapped_column(Text)
    prj_ExpCount: Mapped[int] = mapped_column(Integer, default=0)
    prj_ExpRunCount: Mapped[int] = mapped_column(Integer, default=0)
    prj_MSFilesCount: Mapped[int] = mapped_column(Integer, default=0)
    prj_MSRunTime: Mapped[float] = mapped_column(Float, default=0)
    prj_MSPCEmail: Mapped[str] = mapped_column(Text)
    prj_PaidPrice: Mapped[float | None] = mapped_column(Float, nullable=True)
    prj_ProjectCostMinimum: Mapped[float] = mapped_column(Float, default=0)
    prj_ProjectCostMaximum: Mapped[float | None] = mapped_column(Float, default=None, nullable=True)
    prj_Current_FLAG: Mapped[bool] = mapped_column(Boolean, default=False)
    prj_Billing_ReadyToBill: Mapped[bool] = mapped_column(Boolean, default=False)

    comments: Mapped[List["ProjectComment"]] = relationship(back_populates="project")
    people: Mapped[List["ProjectPerson"]] = relationship(back_populates="project")


class ProjectComment(TIMESTAMP_MIXINS['com'], Base):
    __tablename__ = "project_comment"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"))
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"))
    # com_CreationTS: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # com_ModificationTS: Mapped[datetime] = mapped_column(
    #     DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    # )
    com_Comment: Mapped[str] = mapped_column(Text)

    project: Mapped[Project] = relationship(back_populates="comments")
    person: Mapped[Person] = relationship(back_populates="comments")


PrjPersonTSMixin = make_timestamp_mixin("projper")
class ProjectPerson(PrjPersonTSMixin, Base):
    __tablename__ = "project_person"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"))
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"))

    project: Mapped[Project] = relationship(back_populates="people")
    person: Mapped[Person] = relationship(back_populates="projects")


class LetterOfSupport(TIMESTAMP_MIXINS['los'], Base):
    __tablename__ = "letter_of_support"

    id: Mapped[int] = mapped_column(primary_key=True)
    los_LOSRecNo: Mapped[int] = mapped_column(Integer)
    los_AddedBy: Mapped[str] = mapped_column(Text)
    # los_CreationTS: Mapped[str] = mapped_column(Text)
    # los_ModificationTS: Mapped[datetime] = mapped_column(
    #     DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    # )
    los_ProjectNo: Mapped[int] = mapped_column(Integer)
    los_FileName: Mapped[str] = mapped_column(Text)
    los_PI: Mapped[str] = mapped_column(Text)
    los_PI_Name_Last: Mapped[str] = mapped_column(Text)
    los_PI_Name_First: Mapped[str] = mapped_column(Text)
    los_Institution: Mapped[str] = mapped_column(Text)
    los_Date: Mapped[str] = mapped_column(Text)
    los_Agency: Mapped[str] = mapped_column(Text)
    los_GrantType: Mapped[str] = mapped_column(Text)
    los_LOS_docx: Mapped[bytes] = mapped_column(LargeBinary)
    los_LOS_pdf: Mapped[bytes] = mapped_column(LargeBinary)
    los_Writing: Mapped[str] = mapped_column(Text)
    los_PrelimData: Mapped[str] = mapped_column(Text)
    los_Award_Status: Mapped[str] = mapped_column(Text)
    los_Award_ID: Mapped[int] = mapped_column(Integer)
    los_Award_Name: Mapped[str] = mapped_column(Text)
    los_Award_DateStart: Mapped[str] = mapped_column(Text)
    los_Award_DateEnd: Mapped[str] = mapped_column(Text)
    los_Comment: Mapped[str] = mapped_column(Text)
    los_Award_AnnualDC: Mapped[int] = mapped_column(Integer)
    los_Award_Annual_IDC: Mapped[int] = mapped_column(Integer)
    los_Award_Total: Mapped[float] = mapped_column(Float)
    los_CommentType: Mapped[str] = mapped_column(Text)
    los_ATCReportingYear: Mapped[str] = mapped_column(Text)
    los_FoundCount: Mapped[int] = mapped_column(Integer)
    los_FoundCount_T: Mapped[int] = mapped_column(Integer)
    los_FoundCount_TableT: Mapped[int] = mapped_column(Integer)
    year_: Mapped[str] = mapped_column(Text)
    month_day: Mapped[str] = mapped_column(Text)


# SQLite Engine Factory
def sqlite_engine(db_path="sqlite:///./example.db") -> Engine:
    sqlite3.register_adapter(pd.Timestamp, adapt_timestamp)
    sqlite3.register_converter("TIMESTAMP", convert_timestamp)

    engine = create_engine(
        db_path,
        connect_args={
            "check_same_thread": False,
            "detect_types": sqlite3.PARSE_DECLTYPES,
        },
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        dbapi_connection.set_trace_callback(lambda x: logger.info(x))
        cursor.close()

    return engine



if __name__ == "__main__":
    from sqlalchemy import inspect
    from sqlalchemy.orm import Session

    # Create engine and initialize tables
    engine = sqlite_engine("sqlite:///:memory:")
    #engine = sqlite_engine("file:memdb1?mode=memory&cache=shared")
    Base.metadata.create_all(engine)

    print("\n📦 Tables created in the database:")
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():
        print(f"\n🧱 Table: {table_name}")
        for col in inspector.get_columns(table_name):
            col_type = col['type'].__class__.__name__
            nullable = "NULL" if col['nullable'] else "NOT NULL"
            default = f"DEFAULT {col['default']}" if col['default'] is not None else ""
            print(f"  - {col['name']:30} {col_type:15} {nullable:10} {default}")
