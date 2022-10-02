import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import NullPool
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
    poolclass=NullPool,
)

SessionNoPool = sessionmaker(bind=engine_no_pool)


def main():
    path = Path(sys.argv[1])
    # print(f"processing '{path}'")
    raw_sql = (
        "COPY simulations "
        "FROM STDIN DELIMITER ',' CSV HEADER;"
    )

    # sim_X_deg_BulletName.csv
    pat = re.compile(r"sim_(\d)_deg_(.+)\.csv")
    match = pat.match(path.name)
    if not match:
        raise RuntimeError(
            f"'{path.name}' does not match sim file regex"
        )

    angle = int(match.group(1))
    bullet_name = match[2]
    weapon_name = path.parent.name

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

        if not bullet_id or not weapon_id:
            raise RuntimeError(
                f"invalid bullet or weapon: "
                f"bullet_name={bullet_name}, weapon_name={weapon_name}, "
                f"bullet_id={bullet_id}, weapon_id={weapon_id}"
            )

        # sql = psycopg.sql.SQL(raw_sql).format(path=path)
        c = session.connection().connection.cursor()
        df = pd.read_csv(
            filepath_or_buffer=path,
            dtype={
                "time": np.int16,
                "location_x": np.float32,
                "location_y": np.float32,
                "damage": np.float32,
                "distance": np.float32,
                "velocity": np.float32,
                "energy_transfer": np.float32,
                "power_left": np.float32,
            },
        )
        df["bullet_id"] = bullet_id
        df["weapon_id"] = weapon_id
        df["angle"] = angle
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        with c.copy(raw_sql) as copy:
            copy.write(stream.getvalue())


if __name__ == "__main__":
    main()
