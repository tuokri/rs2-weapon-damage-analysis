from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np

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
    speed: float = -1
    damage: int = -1
    damage_falloff: np.ndarray = np.array([0, 0])
    drag_func: DragFunction = DragFunction.Invalid
    ballistic_coeff: float = -1


@dataclass
class ClassBase:
    name: str = field(hash=True)
    parent: Optional["ClassBase"]

    def __hash__(self) -> int:
        return hash(self.name)

    # TODO: Doesn't work with ProcessPoolExecutor.
    # @lru_cache(maxsize=128, typed=True)
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
                if parent.name == parent.parent.name:
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
                if parent.name == parent.parent.name:
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
        if next_parent.name == next_parent.parent.name:
            return next_parent.name == obj.name
        return next_parent.is_child_of(obj)

    def find_parent(self, parent_name: str) -> "ClassBase":
        if self.name == parent_name:
            return self
        next_parent = self.parent.parent
        if next_parent.name == next_parent.parent.name:
            if parent_name == next_parent.name:
                return next_parent
            else:
                raise ValueError("parent not found")
        return next_parent.find_parent(parent_name)


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
        return self.get_attr("speed", invalid_value=-1)

    def get_speed_uu(self) -> float:
        """Speed in Unreal Units per second (UU/s)."""
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
            if parent.name == parent.parent.name:
                break
        return dmg_fo

    def get_drag_func(self) -> DragFunction:
        return self.get_attr("drag_func", invalid_value=DragFunction.Invalid)

    def get_ballistic_coeff(self) -> float:
        return self.get_attr("ballistic_coeff", invalid_value=-1)


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


@dataclass
class WeaponLoadout(ClassBase):
    """Alternate bullet loadout for a weapon.
    TODO: how to name these? Is is better to just
      use another weapon class with _loadoutX suffix?
    """
    weapon: Weapon
    bullet: Bullet

    def __hash__(self) -> int:
        return super().__hash__()


PROJECTILE = Bullet(
    name="Projectile",
    damage=0,
    speed=0,
    damage_falloff=np.array([0, 1]),
    ballistic_coeff=1.0,
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

    def simulate(self, delta_time: float):
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
    fo_x: np.ndarray = field(init=False)
    fo_y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.bc_inverse = 1 / self.bullet.get_ballistic_coeff()
        # Initial velocity unit vector * muzzle speed.
        v_normalized = self.velocity / np.linalg.norm(self.velocity)
        if np.isnan(v_normalized).any():
            raise RuntimeError("nan encountered in velocity")
        if np.isinf(v_normalized).any():
            raise RuntimeError("inf encountered in velocity")
        if (v_normalized == 0).all():
            raise RuntimeError("velocity must be non-zero")
        bullet_speed = self.bullet.get_speed_uu()
        # print("bullet_speed =", bullet_speed)
        if np.isnan(bullet_speed):
            raise RuntimeError("nan bullet speed")
        if np.isinf(bullet_speed):
            raise RuntimeError("inf bullet speed")
        if bullet_speed <= 0:
            raise RuntimeError("bullet speed <= 0")
        self.velocity = v_normalized * bullet_speed
        # print("velocity =", self.velocity)
        self.fo_x, self.fo_y = interp_dmg_falloff(self.bullet.get_damage_falloff())

    def ef_func(self, speed_squared_uu: float) -> float:
        return np.interp(
            x=speed_squared_uu,
            xp=self.fo_x,
            fp=self.fo_y,
            left=self.fo_y[0],
            right=self.fo_y[-1])

    def calc_drag_coeff(self, mach: float) -> float:
        return str_to_df[self.bullet.get_drag_func()](mach)

    def simulate(self, delta_time: float):
        if delta_time < 0:
            raise RuntimeError("simulation delta time must be >= 0")
        self.flight_time += delta_time
        if (self.velocity == 0).all():
            return
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
                * SCALE_FACTOR * (-1 * (self.velocity / v_size * delta_time)))
        # print("*")
        # print(cd)
        # print(self.bc_inverse)
        # print(v)
        # print(np.square(v))
        # print(self.velocity)
        # print(v_size)
        # FLOAT ZAcceleration = 490.3325 * DeltaTime; // 490.3325 UU/s = 9.806 65 m/s.
        self.velocity[1] -= (490.3325 * delta_time)
        loc_change = self.velocity * delta_time
        prev_loc = self.location.copy()
        self.location += loc_change
        self.distance_traveled_uu += abs(np.linalg.norm(prev_loc - self.location))

    def calc_damage(self) -> float:
        v_size_sq = np.linalg.norm(self.velocity) ** 2
        power_left = v_size_sq / (self.bullet.get_speed_uu() ** 2)
        damage = self.bullet.get_damage() * power_left
        energy_transfer = self.ef_func(v_size_sq)
        # print("power_left      =", power_left)
        # print("energy_transfer =", energy_transfer)
        damage *= energy_transfer
        return damage


def interp_dmg_falloff(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return damage falloff curve x and y sub-arrays with
    zero damage speed data point added added via interpolation.
    TODO: actually, let's not do any interpolation for now.
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
