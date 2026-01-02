import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, make_timestamp_mixin

PplTimestamp = make_timestamp_mixin("ppl")
PrjTimestamp = make_timestamp_mixin("prj")
ComTimestamp = make_timestamp_mixin("com")
ProjectPersonTSMixin = make_timestamp_mixin("projper")


class Person(PplTimestamp, Base):
    __tablename__ = "person"

    id: Mapped[int] = mapped_column(primary_key=True)
    ppl_AddedBy: Mapped[str] = mapped_column(Text)

    ppl_Name_First: Mapped[str] = mapped_column(Text, nullable=False)
    ppl_Name_Last: Mapped[str] = mapped_column(Text, nullable=False)
    ppl_Domain: Mapped[str | None] = mapped_column(Text, nullable=True)
    ppl_Email: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    comments: Mapped[list["ProjectComment"]] = relationship(back_populates="person")
    projects: Mapped[list["ProjectPerson"]] = relationship(back_populates="person")


class ProjectType(str, enum.Enum):
    cprit = "CPRIT"
    rfp = "RFP"
    other = "Other"


class Project(PrjTimestamp, Base):
    __tablename__ = "project"

    id: Mapped[int] = mapped_column(primary_key=True)
    prj_AddedBy: Mapped[str] = mapped_column(Text)
    prj_ProjectTitle: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        info={"ui": {"component": "Text", "label": "Title"}},
    )
    prj_ProjectDescription: Mapped[str | None] = mapped_column(Text, nullable=True)

    prj_ProjectType = mapped_column(
        SAEnum(ProjectType, native_enum=True, validate_strings=True),
        nullable=True,
        info={"ui": {"label": "Project Type", "allowClear": True}},
    )

    prj_PRJ_DisplayID: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_PRJ_DisplayTitle: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_iLabs_RequestName: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_RnD: Mapped[bool] = mapped_column(Boolean, default=False)
    prj_CancerRelevance: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_CPRIT_RFP: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_PI: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Project_LabContact: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Project_LabPersonnel: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_MSPC_Leader: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_MSPC_Personnel_Primary: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_MSPC_Personnel: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_MSPC_Personnel_Analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Services_Type: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Services: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_GrantSupport: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectDomain: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectBackground: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectSuggestions2Customer: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    prj_ProjectSamples: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectCoreTasks: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectQuestions: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Project_SampleType: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Project_FuturePossibilities: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    prj_RequireGelPix: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Experiments_rf: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_RelatedProjects_rf: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Comments_Meeting: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Comments_Billing: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Comments_iLAB: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Status: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        info={
            "ui": {
                "label": "Status",
                "component": "RadioGroup",
                "options": [
                    {"value": "inquiry", "label": "inquiry"},
                    {"value": "consultation", "label": "consultation"},
                    {"value": "waiting", "label": "waiting"},
                    {"value": "processing", "label": "processing"},
                    {"value": "analysis", "label": "analysis"},
                    {"value": "summary", "label": "summary"},
                    {"value": "closed", "label": "closed"},
                    {"value": "hibernate", "label": "hibernate"},
                ],
            }
        },
    )

    prj_Date_Submitted: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    prj_Date_LysatePreparation: Mapped[datetime | None] = mapped_column(
        DateTime, default=None, nullable=True
    )
    prj_Date_MSPreparation: Mapped[datetime | None] = mapped_column(
        DateTime, default=None, nullable=True
    )
    prj_RequireGelPix_Sent: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_RequireGelPix_Confirmed: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Date_Analysis: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True)
    prj_Comments_Specific: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Comments_General: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Date_Review: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True)
    prj_Date_Closed: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True)
    prj_Comments_Review: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_IncludeHandouts: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectCostExplanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Service_365_Analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Service_APMS_Experiments: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Service_APMS_Analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Service_PerBand_Experiments: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Service_PerBand_Analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Service_PTM_Analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Service_DesignHours: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ProjectPrice: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Invoice: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_Invoice_Date: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True)
    prj_Billing_Date: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True)
    prj_PaymentReceived: Mapped[bool] = mapped_column(Boolean, default=False)
    prj_ProjectDomainMSPC: Mapped[str | None] = mapped_column(Text, default=None, nullable=True)
    prj_Project_SampleHandling: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_ExpCount: Mapped[int] = mapped_column(Integer, default=0)
    prj_ExpRunCount: Mapped[int] = mapped_column(Integer, default=0)
    prj_MSFilesCount: Mapped[int] = mapped_column(Integer, default=0)
    prj_MSRunTime: Mapped[float] = mapped_column(Float, default=0)
    prj_MSPCEmail: Mapped[str | None] = mapped_column(Text, nullable=True)
    prj_PaidPrice: Mapped[float | None] = mapped_column(Float, nullable=True)
    prj_ProjectCostMinimum: Mapped[float] = mapped_column(Float, default=0)
    prj_ProjectCostMaximum: Mapped[float | None] = mapped_column(Float, default=None, nullable=True)
    prj_Current_FLAG: Mapped[bool] = mapped_column(Boolean, default=False)
    prj_Billing_ReadyToBill: Mapped[bool] = mapped_column(Boolean, default=False)

    comments: Mapped[list["ProjectComment"]] = relationship(back_populates="project")
    people: Mapped[list["ProjectPerson"]] = relationship(back_populates="project")
    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ProjectComment(ComTimestamp, Base):
    __tablename__ = "project_comment"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"))
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"))
    com_Comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="comments")
    person: Mapped["Person"] = relationship(back_populates="comments")


class ProjectPerson(ProjectPersonTSMixin, Base):
    __tablename__ = "project_person"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"))
    person_id: Mapped[int] = mapped_column(ForeignKey("person.id"))

    project: Mapped["Project"] = relationship(back_populates="people")
    person: Mapped["Person"] = relationship(back_populates="projects")
