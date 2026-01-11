# Models package: split into domain modules but re-exported for compatibility
from ispec.logging import get_logger

from .base import Base, make_timestamp_mixin
from .core import Person, ProjectType, Project, ProjectComment, ProjectPerson
from .experiment import Experiment, ExperimentRun, Job, JobType, JobStatus
from .omics import E2G, GeneContrast, GeneContrastStat, GSEAAnalysis, GSEAResult, PSM
from .files import MSRawFile, ProjectFile, RawFileType, RawFileState, StorageBackend
from .support import LetterOfSupport
from .auth import AuthUser, AuthSession, AuthUserProject, UserRole
from .sync import LegacySyncState
from .engine import sqlite_engine, initialize_db

# Backwards-compatible module-level logger
logger = get_logger(__file__)

__all__ = [
    "Base",
    "make_timestamp_mixin",
    "Person",
    "ProjectType",
    "Project",
    "ProjectComment",
    "ProjectPerson",
    "Experiment",
    "ExperimentRun",
    "E2G",
    "GeneContrast",
    "GeneContrastStat",
    "GSEAAnalysis",
    "GSEAResult",
    "PSM",
    "Job",
    "JobType",
    "JobStatus",
    "MSRawFile",
    "ProjectFile",
    "RawFileType",
    "RawFileState",
    "StorageBackend",
    "LetterOfSupport",
    "AuthUser",
    "AuthSession",
    "AuthUserProject",
    "UserRole",
    "LegacySyncState",
    "sqlite_engine",
    "initialize_db",
    "logger",
]
