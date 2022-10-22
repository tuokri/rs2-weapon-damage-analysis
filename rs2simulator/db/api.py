from functools import lru_cache
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from rs2simulator import db
from rs2simulator.db import models


# TODO: lru_caching here is temporary. Use proper caching with Celery etc. later.

@lru_cache
def get_weapons() -> Iterable[models.Weapon]:
    with db.Session() as session:
        return list(session.scalars(select(models.Weapon)))


@lru_cache
def get_weapon_names() -> Iterable[str]:
    with db.Session() as session:
        return list(session.scalars(select(models.Weapon.name)))


@lru_cache
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


@lru_cache
def get_weapon_id(name: str) -> int:
    with db.Session() as session:
        return session.scalar(
            select(models.Weapon.id).where(
                models.Weapon.name == name
            )
        )


def get_bullet_id(name: str) -> int:
    with db.Session() as session:
        return session.scalar(
            select(models.Bullet.id).where(
                models.Bullet.name == name
            )
        )


@lru_cache
def get_weapon_sim(
        weapon_name: str,
        bullet_name: str,
        angle: float,
) -> pd.DataFrame:
    weapon_id = get_weapon_id(weapon_name)
    bullet_id = get_bullet_id(bullet_name)

    with db.Session() as session:
        stmt = select(models.Simulations).where(
            models.Simulations.weapon_id == weapon_id,
            models.Simulations.bullet_id == bullet_id,
            models.Simulations.angle == angle,
        ).order_by(models.Simulations.time)

        return pd.read_sql(
            sql=stmt,
            con=session.connection(),
            index_col="time",
        )
