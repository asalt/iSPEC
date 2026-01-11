import pytest
from sqlalchemy.orm import sessionmaker

from ispec.db.connect import initialize_db, make_session_factory, sqlite_engine
from ispec.omics.models import OmicsBase


@pytest.fixture(scope="function")
def db_session(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    get_test_session = make_session_factory(engine)
    with get_test_session() as session:
        yield session


@pytest.fixture(scope="function")
def omics_session(tmp_path):
    db_url = f"sqlite:///{tmp_path}/omics.db"
    engine = sqlite_engine(db_url)
    OmicsBase.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
