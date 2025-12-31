from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, LargeBinary, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, make_timestamp_mixin

LosTimestamp = make_timestamp_mixin("los")


class LetterOfSupport(LosTimestamp, Base):
    __tablename__ = "letter_of_support"

    id: Mapped[int] = mapped_column(primary_key=True)
    los_AddedBy: Mapped[str] = mapped_column(Text)
    los_ProjectNo: Mapped[int] = mapped_column(Integer)
    los_FileName: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_PI: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_PI_Name_Last: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_PI_Name_First: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Institution: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Date: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Agency: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_GrantType: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_LOS_docx: Mapped[bytes] = mapped_column(LargeBinary)
    los_LOS_pdf: Mapped[bytes] = mapped_column(LargeBinary)
    los_Writing: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_PrelimData: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Award_Status: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Award_ID: Mapped[int] = mapped_column(Integer)
    los_Award_Name: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Award_DateStart: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Award_DateEnd: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_Award_AnnualDC: Mapped[int] = mapped_column(Integer)
    los_Award_Annual_IDC: Mapped[int] = mapped_column(Integer)
    los_Award_Total: Mapped[float] = mapped_column(Float)
    los_CommentType: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_ATCReportingYear: Mapped[str | None] = mapped_column(Text, nullable=True)
    los_FoundCount: Mapped[int] = mapped_column(Integer)
    los_FoundCount_T: Mapped[int] = mapped_column(Integer)
    los_FoundCount_TableT: Mapped[int] = mapped_column(Integer)
    year_: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True)
    month_day: Mapped[datetime | None] = mapped_column(DateTime, default=None, nullable=True)
