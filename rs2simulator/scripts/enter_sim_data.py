import io
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL
from sqlalchemy import create_engine
from sqlalchemy import make_url
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from rs2simulator.db import models

load_dotenv()

db_url = make_url(os.environ.get("DATABASE_URL"))
engine_no_pool = create_engine(
    url=URL.create(
        drivername="postgresql+psycopg",
        username=db_url.username,
        password=db_url.password,
        host=db_url.host,
        port=db_url.port,
        database=db_url.database,
        query=db_url.query,
    ),
)

SessionNoPool = sessionmaker(bind=engine_no_pool)


def main():
    path = Path(sys.argv[1])
    print(f"processing '{path}'")
    raw_sql = (
        "COPY simulations "
        "FROM STDIN DELIMITER ',' CSV HEADER;"
    )
    bullet_name = str(path.stem.split("_")[-1])
    weapon_name = str(path.parent)
    with SessionNoPool() as session, session.begin():
        bullet_id = session.scalar(
            select(
                models.Bullet.id).where(
                models.Bullet.name == bullet_name
            )
        )
        weapon_id = session.scalar(
            select(
                models.Weapon.id).where(
                models.Weapon.name == weapon_name
            )
        )
        # sql = psycopg.sql.SQL(raw_sql).format(path=path)
        c = session.connection().connection.cursor()
        df = pd.read_csv(path)
        df["bullet_id"] = bullet_id
        df["weapon_id"] = weapon_id
        print(df)
        stream = io.StringIO()
        df.to_csv(stream)
        with c.copy(raw_sql) as copy:
            copy.write(stream.getvalue())


if __name__ == "__main__":
    main()
