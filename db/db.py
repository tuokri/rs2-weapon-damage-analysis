import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import sqlalchemy
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from sqlalchemy import BigInteger
from sqlalchemy import Float
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
    def _repr(self, **fields: Dict[str, Any]) -> str:
        field_strings = []
        at_least_one_attached_attribute = False
        for key, field in fields.items():
            try:
                field_strings.append(f"{key}={field!r}")
            except sqlalchemy.orm.exc.DetachedInstanceError:
                field_strings.append(f"{key}=DetachedInstanceError")
            else:
                at_least_one_attached_attribute = True
        if at_least_one_attached_attribute:
            return f"<{self.__class__.__name__}({','.join(field_strings)})>"
        return f"<{self.__class__.__name__} {id(self)}>"


class DamageFalloff(Base):
    __tablename__ = "damage_falloff"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    bullet_id = mapped_column(
        ForeignKey("bullet.id"),
        onupdate="cascade",
    )
    bullet: Mapped["Bullet"] = relationship(
        back_populates="damage_falloff_curve",
    )
    index: Mapped[int]
    x = mapped_column(BigInteger)
    y = mapped_column(Float)

    def __repr__(self) -> str:
        return self._repr(
            id=self.id,
            index=self.index,
            x=self.x,
            y=self.y,
            bullet=self.bullet,
        )


# TODO: Projectile class needed?
class Bullet(Base):
    __tablename__ = "bullet"

    id = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    name: Mapped[str] = mapped_column(Text)
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("bullet.id"),
        onupdate="cascade",
    )
    parent = relationship(
        "Bullet",
        remote_side=[id],
    )
    ballistic_coeff = mapped_column(Float)
    damage: Mapped[int]
    drag_func: Mapped[int]
    speed: Mapped[int]
    damage_falloff_curve: Mapped[List[DamageFalloff]] = relationship(
        back_populates="bullet",
    )


class Weapon(Base):
    __tablename__ = "weapon"

    id = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True, )
    name: Mapped[str] = mapped_column(Text)
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
    ammo_loadouts: Mapped[List["AmmoLoadout"]] = relationship()


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
