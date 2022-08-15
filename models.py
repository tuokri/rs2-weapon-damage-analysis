import math
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from functools import lru_cache
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

from drag import drag_g1
from drag import drag_g7

SCALE_FACTOR_INVERSE = 0.065618
SCALE_FACTOR = 15.24


class DragFunction(Enum):
    Invalid = ""
    G1 = "RODF_G1"
    G7 = "RODF_G7"


@dataclass
class ParseResult:
    class_name: str = ""
    parent_name: str = ""


@dataclass
class WeaponParseResult(ParseResult):
    bullet_name: str = ""
    instant_damage: int = -1
    pre_fire_length: int = -1


@dataclass
class BulletParseResult(ParseResult):
    speed: float = math.inf
    damage: int = -1
    damage_falloff: np.ndarray = np.array([0, 0])
    drag_func: DragFunction = DragFunction.Invalid
    ballistic_coeff: float = math.inf


@dataclass
class ClassBase:
    name: str = field(hash=True)
    parent: Optional["ClassBase"]

    def __hash__(self) -> int:
        return hash(self.name)

    @lru_cache(maxsize=128, typed=True)
    def get_attr(self,
                 attr_name: str,
                 invalid_value: Optional[Any] = None) -> Any:
        if invalid_value is not None:
            attr = getattr(self, attr_name)
            if attr != invalid_value:
                return attr
            parent = self.parent
            while attr == invalid_value:
                attr = getattr(parent, attr_name)
                parent = parent.parent
                if parent is parent.parent:
                    attr = getattr(parent.parent, attr_name)
                    break
            return attr
        else:
            attr = getattr(self, attr_name)
            if attr:
                return attr
            parent = self.parent
            while not attr:
                attr = getattr(parent, attr_name)
                parent = parent.parent
                if parent is parent.parent:
                    attr = getattr(parent.parent, attr_name)
                    break
            return attr

    def is_child_of(self, obj: "ClassBase") -> bool:
        if not self.parent:
            return False
        if self.name == obj.name:
            return True
        parent = self.parent
        if parent.name == obj.name:
            return True
        next_parent = parent.parent
        if not next_parent:
            return False
        return next_parent.is_child_of(obj)


@dataclass
class Bullet(ClassBase):
    parent: Optional["Bullet"]
    speed: float
    damage: int
    damage_falloff: np.ndarray
    drag_func: DragFunction
    ballistic_coeff: float

    def __hash__(self) -> int:
        return super().__hash__()

    def get_speed(self) -> float:
        """Speed (muzzle velocity) in m/s."""
        return self.get_attr("speed", invalid_value=math.inf)

    def get_speed_uu(self) -> float:
        """Speed Unreal Units per second (UU/s)."""
        return self.get_speed() * 50

    def get_damage(self) -> int:
        return self.get_attr("damage", invalid_value=-1)

    def get_damage_falloff(self) -> np.ndarray:
        dmg_fo = self.damage_falloff
        if (dmg_fo > 0).any():
            return dmg_fo
        parent = self.parent
        while not (dmg_fo > 0).any():
            dmg_fo = parent.damage_falloff
            parent = parent.parent
            if parent is parent.parent:
                break
        return dmg_fo

    def get_drag_func(self) -> DragFunction:
        return self.get_attr("drag_func", invalid_value=DragFunction.Invalid)

    def get_ballistic_coeff(self) -> float:
        return self.get_attr("ballistic_coeff", invalid_value=math.inf)


@dataclass
class Weapon(ClassBase):
    parent: Optional["Weapon"]
    bullet: Optional[Bullet]
    instant_damage: int
    pre_fire_length: int

    def __hash__(self) -> int:
        return super().__hash__()

    def get_bullet(self) -> Bullet:
        return self.get_attr("bullet")

    def get_instant_damage(self) -> int:
        return self.get_attr("instant_damage", invalid_value=-1)

    def get_pre_fire_length(self) -> int:
        return self.get_attr("pre_fire_length", invalid_value=-1)


PROJECTILE = Bullet(
    name="Projectile",
    damage=0,
    speed=0,
    damage_falloff=np.array([0, 1]),
    ballistic_coeff=0,
    drag_func=DragFunction.G1,
    parent=None,
)
PROJECTILE.parent = PROJECTILE
WEAPON = Weapon(
    name="Weapon",
    bullet=PROJECTILE,
    parent=None,
    pre_fire_length=50,
    instant_damage=0
)
WEAPON.parent = WEAPON

str_to_df = {
    DragFunction.G1: drag_g1,
    DragFunction.G7: drag_g7,
}


@dataclass
class WeaponSimulation:
    weapon: Weapon
    velocity: np.ndarray = np.array([1, 0], dtype=np.longfloat)
    location: np.ndarray = np.array([0, 1], dtype=np.longfloat)
    bullet: Bullet = field(init=False)
    sim: "BulletSimulation" = field(init=False)

    def __post_init__(self):
        self.bullet = self.weapon.get_bullet()
        self.sim = BulletSimulation(
            bullet=self.bullet,
            velocity=self.velocity.copy(),
            location=self.location.copy(),
        )

    @property
    def distance_traveled_uu(self) -> float:
        return self.sim.distance_traveled_uu

    @property
    def distance_traveled_m(self) -> float:
        return self.distance_traveled_uu / 50

    @property
    def flight_time(self) -> float:
        return self.sim.flight_time

    @property
    def ef_func(self) -> Callable:
        return self.sim.ef_func

    def calc_drag_coeff(self, mach: float) -> float:
        return self.sim.calc_drag_coeff(mach)

    def simulate(self, delta_time: np.longfloat):
        self.sim.simulate(delta_time)
        self.velocity = self.sim.velocity.copy()
        self.location = self.sim.location.copy()

    def calc_damage(self) -> float:
        return self.sim.calc_damage()


@dataclass
class BulletSimulation:
    bullet: Bullet
    flight_time: float = 0
    bc_inverse: float = 0
    distance_traveled_uu: float = 0
    velocity: np.ndarray = np.array([1, 0], dtype=np.longfloat)
    location: np.ndarray = np.array([0, 1], dtype=np.longfloat)
    ef_func: Callable = field(init=False)

    def __post_init__(self):
        self.bc_inverse = self.bullet.get_ballistic_coeff() / 1
        # Initial velocity unit vector * muzzle speed.
        v_normalized = self.velocity / np.linalg.norm(self.velocity)
        self.velocity = v_normalized * self.bullet.get_speed_uu()
        fo_x, fo_y = interp_dmg_falloff(self.bullet.get_damage_falloff())
        self.ef_func = interp1d(
            fo_x, fo_y, kind="linear",
            # fill_value="extrapolate", bounds_error=False)
            fill_value=(fo_y[0], fo_y[-1]), bounds_error=False)

    def calc_drag_coeff(self, mach: float) -> float:
        return str_to_df[self.bullet.get_drag_func()](mach)

    def simulate(self, delta_time: np.longfloat):
        if delta_time < 0:
            raise RuntimeError("simulation delta time must be >= 0")
        self.flight_time += delta_time
        # if (self.velocity == 0).all():
        #     return
        # FLOAT V = Velocity.Size() * UCONST_ScaleFactorInverse.
        v_size = np.linalg.norm(self.velocity)
        v = v_size * SCALE_FACTOR_INVERSE
        mach = v * 0.0008958245617
        cd = self.calc_drag_coeff(mach)
        # FVector AddVelocity = 0.00020874137882624 * (CD * BCInverse)
        # * Square(V) * UCONST_ScaleFactor * (-1 * VelocityNormal * DeltaTime);
        self.velocity += (
                0.00020874137882624
                * (cd * self.bc_inverse) * np.square(v)
                * SCALE_FACTOR * (-1 * (self.velocity / v_size)))
        # FLOAT ZAcceleration = 490.3325 * DeltaTime; // 490.3325 UU/s = 9.806 65 m/s.
        self.velocity[1] -= (490.3325 * delta_time)
        loc_change = self.velocity * delta_time
        prev_loc = self.location.copy()
        self.location += loc_change
        self.distance_traveled_uu += abs(np.linalg.norm(prev_loc - self.location))

    def calc_damage(self) -> float:
        # v_size = np.linalg.norm(self.velocity) / 50  # m/s.
        v_size_sq = np.linalg.norm(self.velocity) ** 2
        # print("speed =", v_size, "m/s")
        power_left = v_size_sq / (self.bullet.get_speed_uu() ** 2)
        damage = self.bullet.get_damage() * power_left
        energy_transfer = self.ef_func(v_size_sq)
        print("power_left      =", power_left)
        print("energy_transfer =", energy_transfer)
        damage *= energy_transfer
        return damage


def interp_dmg_falloff(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return damage falloff curve x and y sub-arrays with
    zero damage speed data point added added via interpolation.
    """
    harr = np.hsplit(arr, 2)
    x = harr[0].ravel()
    y = harr[1].ravel()
    # f_xtoy = interp1d(x, y, fill_value="extrapolate", kind="linear")
    # try:
    #     f_ytox = interp1d(y, x, fill_value="extrapolate", kind="linear")
    # except ValueError:
    #     return x, y
    # zero_dmg_speed = f_ytox(1)
    # x = np.insert(x, 0, zero_dmg_speed)
    # y = np.insert(y, 0, 1)
    return x, y
