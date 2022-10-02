import functools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import sqlalchemy
from sqlalchemy import BigInteger
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import SmallInteger
from sqlalchemy import Text
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class PrettyReprMixin:
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


class BaseModel(PrettyReprMixin, DeclarativeBase):
    __abstract__ = True


_AutomapBase = automap_base()


class AutomapModel(PrettyReprMixin, _AutomapBase):
    __abstract__ = True


class DamageFalloff(BaseModel):
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
            bullet_name=self.bullet.name,
        )


# TODO: Projectile class needed?
class Bullet(BaseModel):
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

    def __repr__(self) -> str:
        r = functools.partial(
            self._repr,
            id=self.id,
            name=self.name,
            ballistic_coeff=self.ballistic_coeff,
            damage=self.damage,
            drag_func=self.drag_func,
            speed=self.speed,
            damage_falloff_curve=self.damage_falloff_curve,
        )

        if self.parent is not None:
            return r(parent_name=self.parent.name)
        else:
            return r(parent=None)


class Weapon(BaseModel):
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

    def __repr__(self) -> str:
        return self._repr(
            id=self.id,
            name=self.name,
            display_name=self.display_name,
            short_display_name=self.short_display_name,
            pre_fire_length=self.pre_fire_length,
            ammo_loadouts=self.ammo_loadouts,
            parent_name=self.parent.name,
        )


class AmmoLoadout(BaseModel):
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


class Simulations(AutomapModel):
    """TimescaleDB hypertable."""
    __tablename__ = "simulations"

    time: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        primary_key=True,
    )
    location_x: Mapped[float]
    location_y: Mapped[float]
    damage: Mapped[float] = mapped_column(Float)
    distance: Mapped[float]
    velocity: Mapped[float]
    energy_transfer: Mapped[float]
    power_left: Mapped[float]
    angle: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        primary_key=True,
    )
    bullet_id: Mapped[int] = mapped_column(
        ForeignKey("bullet.id"),
        nullable=False,
        primary_key=True,
    )
    weapon_id: Mapped[int] = mapped_column(
        ForeignKey("weapon.id"),
        nullable=False,
        primary_key=True,
    )

    # __mapper_args__ = {
    #     "primary_key": [
    #         time,
    #         bullet_id,
    #         weapon_id,
    #         angle,
    #     ]
    # }
