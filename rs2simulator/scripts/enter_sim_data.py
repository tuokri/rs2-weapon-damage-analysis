# Copyright (C) 2022-2025 Tuomo Kriikkula
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Enter simulation data into a PostgreSQL database.
NOTE: This script is not meant to be executed manually, but rather
by rs2simulator/main.py, using the -e argument.
"""

import argparse
import atexit
import io
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict

import logbook
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from psycopg_pool import NullConnectionPool
from sqlalchemy import Engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

logger = logbook.Logger("enter_sim_data")

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    raise RuntimeError("no database URL")

pool: NullConnectionPool
engine_no_pool: Engine
SessionNoPool: sessionmaker

csv_header = (
    "time,location_x,location_y,damage,distance,"
    "velocity,energy_transfer,power_left,angle,bullet_id,weapon_id"
)


def main(
        data_dir: Path,
        weapon_ids: Dict[str, int],
        bullet_ids: Dict[str, int],
) -> None:
    start = time.perf_counter()

    logger.info("handling '{f}'", f=data_dir)

    stop_csv = 0.0
    start_copy = 0.0
    stop_copy = 0.0

    raw_sql = (
        "COPY simulations "
        "FROM STDIN DELIMITER ',' CSV HEADER;"
    )

    csv_paths = list(data_dir.glob("*.csv"))

    # sim_X_deg_BulletName.csv
    sim_file_pat = re.compile(r"sim_(\d)_deg_(.+)\.csv")
    weapon_name = data_dir.name
    weapon_id = weapon_ids[weapon_name]

    with SessionNoPool(autoflush=False) as session, session.begin():
        cursor = session.connection().connection.cursor()

        start_csv = time.perf_counter()
        with io.BytesIO() as stream:
            stream.write(f"{csv_header}\n".encode())
            for path in csv_paths:
                match = sim_file_pat.match(path.name)
                if not match:
                    raise RuntimeError(
                        f"{path.name} failed to match sim file pattern")
                bullet_name = match.group(2)
                bullet_id = bullet_ids[bullet_name]
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
                    path_or_buf=stream,
                    header=False,
                    index=False,
                    mode="a",
                    encoding=None,
                    lineterminator="\n",
                )
            stop_csv = time.perf_counter()

            start_copy = time.perf_counter()
            with cursor.copy(raw_sql) as copy:
                copy.write(stream.getvalue())
            stop_copy = time.perf_counter()

    stop = time.perf_counter()

    total = stop - start
    csv_time = stop_csv - start_csv
    copy_time = stop_copy - start_copy
    # select_time = stop_select - start_select

    logger.info(
        f"total={total}, "
        f"csv={csv_time / total:.2%}, "
        f"copy={copy_time / total:.2%}, "
        # f"select={select_time / total:.2%}"
    )


class KeyValueAction(argparse.Action):
    def __call__(self, parser, namespace,
                 values, option_string=None):
        setattr(namespace, self.dest, {})

        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = int(value)


if __name__ == "__main__":
    _pid = os.getpid()
    _format_string = (
        f"[{{record.time:%Y-%m-%d %H:%M:%S.%f%z}}] {{record.level_name}}: "
        f"{{record.channel}}: {_pid}: {{record.message}}")
    _setup = logbook.NestedSetup([
        logbook.NullHandler(),
        logbook.StreamHandler(sys.stdout, format_string=_format_string),
    ])

    _ap = argparse.ArgumentParser()
    _ap.add_argument(
        "data_dir",
        help="weapon data directory containing rs2simlib-generated CSV data",
    )
    # noinspection PyTypeChecker
    _ap.add_argument(
        "-b",
        "--bullet-ids",
        help="bullet name to id key-value pairs in BulletName=DatabaseID format",
        nargs="*",
        action=KeyValueAction,
        metavar="KEY=VALUE",
        required=True,
    )
    # noinspection PyTypeChecker
    _ap.add_argument(
        "-w",
        "--weapon-ids",
        help="weapon name to id key-value pairs in WeaponName=DatabaseID format",
        nargs="*",
        action=KeyValueAction,
        metavar="KEY=VALUE",
        required=True,
    )
    _ap.add_argument(
        "--no-prepared-statements",
        action="store_true",
        help="disable Psycopg prepared statements",
    )

    _args = _ap.parse_args()
    _no_preps = _args.no_prepared_statements
    _prep_thresh = None if _no_preps else 5

    pool = NullConnectionPool(
        conninfo=db_url,
    )
    atexit.register(pool.close)
    engine_no_pool = create_engine(
        url="postgresql+psycopg://",
        creator=pool.getconn,
        connect_args={"prepare_threshold": None},
    )
    SessionNoPool = sessionmaker(bind=engine_no_pool)

    with _setup.applicationbound():
        try:
            _data_dir = Path(_args.data_dir)
            main(
                data_dir=_data_dir,
                weapon_ids=_args.weapon_ids,
                bullet_ids=_args.bullet_ids,
            )
        except Exception as e:
            logger.error(
                "failed for: '{wep}': {e}",
                wep=_data_dir.name,
                e=e,
            )
            # logger.exception(e)
            raise SystemExit(1)
