from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from rs2simulator import db
from rs2simulator.db import models


def get_weapons() -> Iterable[models.Weapon]:
    with db.Session() as session:
        return list(session.scalars(select(models.Weapon)))


def get_weapon_names() -> Iterable[str]:
    with db.Session() as session:
        return list(session.scalars(select(models.Weapon.name)))


def get_weapon(name: str) -> models.Weapon:
    with db.Session() as session:
        return session.scalar(
            select(models.Weapon).where(
                models.Weapon.name == name
            ).options(
                selectinload(
                    models.Weapon.ammo_loadouts
                ).selectinload(
                    models.AmmoLoadout.bullet
                ).selectinload(
                    models.Bullet.damage_falloff_curve
                ).selectinload(
                    models.DamageFalloff.bullet
                )
            )
        )
