from typing import Iterable

from sqlalchemy import select

from rs2simulator import db
from rs2simulator.db import models


def get_weapons() -> Iterable[models.Weapon]:
    with db.Session() as session:
        return list(session.scalars(select(models.Weapon)))


def get_weapon_names() -> Iterable[str]:
    with db.Session() as session:
        return list(session.scalars(select(models.Weapon.name)))
