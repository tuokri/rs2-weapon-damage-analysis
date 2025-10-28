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

import functools
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt
import sqlalchemy
from sqlalchemy import BigInteger
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import SmallInteger
from sqlalchemy import Text
from sqlalchemy import inspect
from sqlalchemy.ext.automap import AutomapBase
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class PrettyReprMixin:
    def _repr(self, **fields: Any) -> str:
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

    def to_dict(self) -> dict:
        d = {
            key: getattr(self, key)
            for key in self.__mapper__.c.keys()
            if not key.startswith("_")

        }
        d_hybrid = {
            key: getattr(self, key)
            for key, prop in inspect(self.__class__).all_orm_descriptors.items()
            if isinstance(prop, hybrid_property)
        }

        d.update(d_hybrid)
        return d


_AutomapBase: AutomapBase = automap_base()


class AutomapModel(
    PrettyReprMixin,
    _AutomapBase,  # type: ignore[valid-type, misc]
):
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
    # parent = relationship(
    #     "Bullet",
    #     remote_side=[id],
    # )
    ballistic_coeff = mapped_column(Float)
    damage: Mapped[int]
    drag_func: Mapped[int]
    speed: Mapped[float]
    damage_falloff_curve: Mapped[List[DamageFalloff]] = relationship(
        back_populates="bullet",
        order_by="asc(DamageFalloff.index)",
    )

    def dmg_falloff_np_tuple(self) -> Tuple[npt.NDArray, npt.NDArray]:
        x = []
        y = []
        for fo in self.damage_falloff_curve:
            x.append(fo.x)
            y.append(fo.y)
        return np.array(x), np.array(y)

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

        # if self.parent is not None:
        #     return r(parent_name=self.parent.name)
        # else:
        #     return r(parent=None)
        return r(parent_id=self.parent_id)


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
    # parent = relationship(
    #     "Weapon",
    #     remote_side=[id],
    # )
    display_name: Mapped[Optional[str]]
    short_display_name: Mapped[Optional[str]]
    pre_fire_length: Mapped[int]
    ammo_loadouts: Mapped[List["AmmoLoadout"]] = relationship(
        "AmmoLoadout",
    )

    def __repr__(self) -> str:
        return self._repr(
            id=self.id,
            name=self.name,
            display_name=self.display_name,
            short_display_name=self.short_display_name,
            pre_fire_length=self.pre_fire_length,
            ammo_loadouts=self.ammo_loadouts,
            # parent_name=self.parent.name,
            parent_id=self.parent_id,
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
    bullet: Mapped["Bullet"] = relationship()

    # spread: Mapped[float]
    # num_projectiles: Mapped[int]
    # damage_type: Mapped[int] = mapped_column(ForeignKey(damage_type.id))

    def __repr__(self) -> str:
        return self._repr(
            id=self.id,
            weapon_id=self.weapon_id,
            bullet_id=self.bullet_id,
            instant_damage=self.instant_damage,
            bullet=self.bullet,
        )


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
