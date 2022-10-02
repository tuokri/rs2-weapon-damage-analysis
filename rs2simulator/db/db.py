import os
from pathlib import Path
from typing import Any
from typing import Optional
from typing import TypeVar

from psycopg_pool import ConnectionPool
from sqlalchemy import Engine
from sqlalchemy import create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import DropTable

from rs2simulator.db.models import AutomapModel
from rs2simulator.db.models import BaseModel

# TODO: pgbouncer for prod?
_pool: Optional[ConnectionPool] = None
_engine: Optional[Engine] = None

_S = TypeVar("_S", bound="_ORMSession")


def engine() -> Engine:
    global _pool
    global _engine

    if _pool is None:
        db_url = os.environ.get("DATABASE_URL")
        _pool = ConnectionPool(
            conninfo=db_url,
        )

    if _engine is None:
        protocol = "postgresql+psycopg://"
        _engine = create_engine(
            url=protocol,
            creator=_pool.getconn,
        )

    return _engine


# noinspection PyPep8Naming
class session_maker(sessionmaker):
    def __call__(self, **local_kw: Any) -> _S:
        local_kw["bind"] = engine()
        return super().__call__(**local_kw)


Session = session_maker()


def drop_create_all(db_engine: Optional[Engine] = None):
    @compiles(DropTable, "postgresql")
    def _compile_drop_table(element, compiler):
        return f"{compiler.visit_drop_table(element)} CASCADE"

    if db_engine is None:
        db_engine = engine()

    pool_dispose(db_engine)

    BaseModel.metadata.drop_all(db_engine)
    BaseModel.metadata.create_all(db_engine)
    with Session() as s, s.begin():
        _timescale_sql = (
                Path(__file__).parent / "timescale.sql").read_text()
        _c = s.connection().connection.cursor()
        _c.execute(_timescale_sql)
    AutomapModel.prepare(db_engine, reflect=True)


def pool_dispose(db_engine: Engine):
    db_engine.dispose(close=False)
