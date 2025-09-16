import pytest
from ispec.db.connect import make_session_factory, sqlite_engine, initialize_db


@pytest.fixture(scope="function")
def db_session(tmp_path):
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = sqlite_engine(db_url)
    initialize_db(engine)
    get_test_session = make_session_factory(engine)
    with get_test_session() as session:
        yield session
