import os
from pathlib import Path

from psycopg_pool import ConnectionPool
from sqlalchemy import Engine
from sqlalchemy import create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import DropTable

from rs2simulator.db.models import AutomapModel
from rs2simulator.db.models import BaseModel

_DATABASE_URL = os.environ.get("DATABASE_URL")
_PROTOCOL_URL = "postgresql+psycopg://"

# TODO: pgbouncer for prod?
pool = ConnectionPool(
    conninfo=_DATABASE_URL,
)

engine = create_engine(
    url=_PROTOCOL_URL,
    creator=pool.getconn,
)

Session = sessionmaker(bind=engine)


def drop_create_all(db_engine: Engine):
    @compiles(DropTable, "postgresql")
    def _compile_drop_table(element, compiler):
        return f"{compiler.visit_drop_table(element)} CASCADE"

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
