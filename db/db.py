import os
from typing import List
from typing import Optional

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import Text
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bullet_name: Mapped[str] = mapped_column(ForeignKey("bullet.name"))
    index: Mapped[int]
    x: Mapped[int]
    y: Mapped[int]


# TODO: Projectile class needed? Integer primary key?
class Bullet(Base):
    __tablename__ = "bullet"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    parent_name: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("bullet.name"))
    parent = relationship(
        "Bullet",
        remote_side=name,
        post_update=True)
    ballistic_coeff: Mapped[float]
    damage: Mapped[int]
    drag_func: Mapped[int]  # TODO: drag function table?
    speed: Mapped[int]
    damage_falloff_curve: Mapped[List[DamageFalloff]] = relationship()


class Weapon(Base):
    __tablename__ = "weapon"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    parent_name: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("weapon.name"))
    parent = relationship(
        "Weapon",
        remote_side=name,
        post_update=True)
    display_name: Mapped[Optional[str]]
    short_display_name: Mapped[Optional[str]]
    pre_fire_length: Mapped[int]


class AmmoLoadout(Base):
    __tablename__ = "ammo_loadout"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    weapon_name: Mapped[str] = mapped_column(ForeignKey("weapon.name"))
    bullet_name: Mapped[str] = mapped_column(ForeignKey("bullet.name"))
    instant_damage: Mapped[int]
    # spread: Mapped[float]
    # num_projectiles: Mapped[int]
    # damage_type: Mapped[int] = mapped_column(ForeignKey(damage_type.id))


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
