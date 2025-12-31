# Models package: split into domain modules but re-exported for compatibility
from ispec.logging import get_logger

from .base import Base, make_timestamp_mixin
from .core import Person, ProjectType, Project, ProjectComment, ProjectPerson
from .experiment import Experiment, ExperimentRun, E2G, Job, JobType, JobStatus
from .omics import PSM
from .files import MSRawFile, RawFileType, RawFileState, StorageBackend
from .support import LetterOfSupport
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
    "PSM",
    "Job",
    "JobType",
    "JobStatus",
    "MSRawFile",
    "RawFileType",
    "RawFileState",
    "StorageBackend",
    "LetterOfSupport",
    "sqlite_engine",
    "initialize_db",
    "logger",
]
