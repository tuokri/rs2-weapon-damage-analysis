import io
import os
import re
import sys
import time
from pathlib import Path

import logbook
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

logger = logbook.Logger("enter_sim_data")

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

csv_header = (
    "time,location_x,location_y,damage,distance,"
    "velocity,energy_transfer,power_left,angle,bullet_id,weapon_id"
)


def main(data_dir: Path):
    start = time.perf_counter()

    logger.info("handling '{f}'", f=data_dir)

    start_csv = 0
    stop_csv = 0
    start_copy = 0
    stop_copy = 0
    start_select = 0
    stop_select = 0

    raw_sql = (
        "COPY simulations "
        "FROM STDIN DELIMITER ',' CSV HEADER;"
    )

    csv_paths = list(data_dir.glob("*.csv"))

    # sim_X_deg_BulletName.csv
    any_csv = csv_paths[0]
    pat = re.compile(r"sim_(\d)_deg_(.+)\.csv")
    match = pat.match(any_csv.name)
    if not match:
        raise RuntimeError(
            f"'{any_csv.name}' does not match sim file regex"
        )

    bullet_name = match.group(2)
    weapon_name = data_dir.name

    with SessionNoPool(autoflush=False) as session, session.begin():
        # TODO: optimize this further by doing a single select
        #  in the parent process and pass the needed IDs as
        #  arguments to each process.
        start_select = time.perf_counter()
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
        stop_select = time.perf_counter()

        if not all((bullet_id, weapon_id, bullet_name, weapon_name)):
            raise RuntimeError(
                f"invalid bullet or weapon: "
                f"bullet_name={bullet_name}, weapon_name={weapon_name}, "
                f"bullet_id={bullet_id}, weapon_id={weapon_id}"
            )

        # sql = psycopg.sql.SQL(raw_sql).format(path=path)
        c = session.connection().connection.cursor()

        start_csv = time.perf_counter()
        with io.BytesIO() as stream:
            stream.write(f"{csv_header}\n".encode())
            for path in csv_paths:
                match = pat.match(path.name)
                angle = int(match.group(1))
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
                df["angle"] = angle
                df["bullet_id"] = bullet_id
                df["weapon_id"] = weapon_id
                df.to_csv(
                    stream,
                    index=False,
                    header=False,
                    encoding=None,
                    line_terminator="\n",
                )
            stop_csv = time.perf_counter()

            start_copy = time.perf_counter()
            with c.copy(raw_sql) as copy:
                copy.write(stream.getvalue())
            stop_copy = time.perf_counter()

    stop = time.perf_counter()

    total = stop - start
    csv_time = stop_csv - start_csv
    copy_time = stop_copy - start_copy
    select_time = stop_select - start_select

    logger.info(
        f"total={total}, "
        f"csv={csv_time / total:.2%}, "
        f"copy={copy_time / total:.2%}, "
        f"select={select_time / total:.2%}"
    )


if __name__ == "__main__":
    _pid = os.getpid()
    _format_string = (
        f"[{{record.time:%Y-%m-%d %H:%M:%S.%f%z}}] {{record.level_name}}: "
        f"{{record.channel}}: {_pid}: {{record.message}}")
    _setup = logbook.NestedSetup([
        logbook.NullHandler(),
        logbook.StreamHandler(sys.stdout, format_string=_format_string),
    ])

    with _setup.applicationbound():
        try:
            _data_dir = Path(sys.argv[1])
            main(_data_dir)
        except Exception as e:
            logger.error(
                "failed for: '{wep}': {e}",
                wep=_data_dir.name,
                e=e,
            )
            # logger.exception(e)
            raise SystemExit(1)
