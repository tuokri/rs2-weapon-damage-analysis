import os
from typing import List
from typing import Optional

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker

load_dotenv()

pool = ConnectionPool(
    conninfo=os.environ.get("DATABASE_URL"),
)

engine = create_engine(
    url="postgresql+psycopg://",
    creator=pool.getconn,
)

Session = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class DamageFalloff(Base):
    __tablename__ = "damage_falloff"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    bullet_name: Mapped[int] = mapped_column(
        ForeignKey("bullet.id"),
        onupdate="cascade",
    )
    index: Mapped[int]
    x: Mapped[int]
    y: Mapped[int]


# TODO: Projectile class needed?
class Bullet(Base):
    __tablename__ = "bullet"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    name: Mapped[str]
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("bullet.id"),
        onupdate="cascade",
    )
    parent = relationship(
        "Bullet",
        remote_side=[id],
    )
    ballistic_coeff: Mapped[float]
    damage: Mapped[int]
    drag_func: Mapped[int]  # TODO: drag function table?
    speed: Mapped[int]
    damage_falloff_curve: Mapped[List[DamageFalloff]] = relationship()


class Weapon(Base):
    __tablename__ = "weapon"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True, )
    name: Mapped[str]
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("weapon.id"),
        onupdate="cascade",
    )
    parent = relationship(
        "Weapon",
        remote_side=[id],
    )
    display_name: Mapped[Optional[str]]
    short_display_name: Mapped[Optional[str]]
    pre_fire_length: Mapped[int]


class AmmoLoadout(Base):
    __tablename__ = "ammo_loadout"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    weapon_id: Mapped[int] = mapped_column(
        ForeignKey("weapon.id"),
        onupdate="cascade",
    )
    bullet_id: Mapped[int] = mapped_column(
        ForeignKey("bullet.id"),
        onupdate="cascade",
    )
    instant_damage: Mapped[int]
    # spread: Mapped[float]
    # num_projectiles: Mapped[int]
    # damage_type: Mapped[int] = mapped_column(ForeignKey(damage_type.id))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
