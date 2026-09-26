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

import shutil
import subprocess
import sys
from typing import Any
from typing import Generator

import pytest

from . import data_dir
from . import main_py

# TODO: strategy for bringing in enough testing data!

uscript_dir = data_dir / "UnrealScript"
localization_dir = data_dir / "Localization"
out_dir = data_dir / "out"
root_dir_arg = f"--root-dir={out_dir}"


@pytest.fixture(scope="module")
def make_out_dir() -> Generator[None, Any, None]:
    out_dir.mkdir(exist_ok=True, parents=True)
    yield
    shutil.rmtree(out_dir)


def test_help(make_out_dir) -> None:
    subprocess.check_call(
        [
            sys.executable,
            main_py,
            "--help",
            root_dir_arg,
        ],
    )


@pytest.mark.dependency()
def test_parse_unrealscript(make_out_dir) -> None:
    ret = subprocess.check_call(
        [
            sys.executable,
            main_py,
            "--parse-src",
            str(uscript_dir),
            root_dir_arg,
        ]
    )
    assert ret == 0
    assert (out_dir / "weapon_classes.pickle").exists()
    assert (out_dir / "bullet_classes.pickle").exists()
    assert (out_dir / "weapons.json").exists()
    assert (out_dir / "weapons_readable.json").exists()
    assert (out_dir / "bullets.json").exists()
    assert (out_dir / "bullets_readable.json").exists()


@pytest.mark.dependency(depends=[test_parse_unrealscript.__name__])
def test_parse_localization(make_out_dir) -> None:
    ret = subprocess.check_call(
        [
            sys.executable,
            main_py,
            "--parse-localization",
            str(localization_dir / "Test.int"),
            root_dir_arg,
        ]
    )
    assert ret == 0


@pytest.mark.dependency(depends=[test_parse_localization.__name__])
def test_run_simulations(make_out_dir) -> None:
    ret = subprocess.check_call(
        [
            sys.executable,
            main_py,
            "--simulate",
            str(out_dir / "weapon_classes.pickle"),
            root_dir_arg,
        ]
    )
    assert ret == 0
