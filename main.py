import datetime
import json
import math
import re
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import MutableMapping
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from requests.structures import CaseInsensitiveDict
from scipy.interpolate import interp1d

from drag import drag_g1
from drag import drag_g7

# TODO: take as argument?
SRC_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\Development"

SCALE_FACTOR_INVERSE = 0.065618
SCALE_FACTOR = 15.24

INSTANT_DAMAGE_PATTERN = re.compile(
    r"^\s*InstantHitDamage\(\d+\)\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
PRE_FIRE_PATTERN = re.compile(
    r"^\s*PreFireTraceLength\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
DRAG_FUNC_PATTERN = re.compile(
    r"^\s*DragFunction\s*=\s*([\w\d_]+).*$",
    flags=re.IGNORECASE,
)
BALLISTIC_COEFF_PATTERN = re.compile(
    r"^\s*BallisticCoefficient\s*=\s*([\d.]+).*$",
    flags=re.IGNORECASE,
)
SPEED_PATTERN = re.compile(
    r"^\s*Speed\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
DAMAGE_PATTERN = re.compile(
    r"^\s*Damage\s*=\s*(\d+).*$",
    flags=re.IGNORECASE,
)
FALLOFF_PATTERN = re.compile(
    r"^\s*VelocityDamageFalloffCurve\s*=\s*\(Points\s*=\s*(\(.*\))\).*$",
    flags=re.IGNORECASE,
)
CLASS_PATTERN = re.compile(
    r"^\s*class\s+([\w]+)\s+extends\s+([\w\d_]+)\s*.*[;\n\s]+$",
    flags=re.IGNORECASE,
)
WEAPON_BULLET_PATTERN = re.compile(
    r"^\s*WeaponProjectiles\(\d+\)\s*=\s*class\s*'(.*)'.*$",
    flags=re.IGNORECASE,
)


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


def is_comment(line: str) -> bool:
    # Don't bother with multi-line comments for now.
    return line.lstrip().startswith("//")


def rstrip_comment(line: str) -> str:
    if "//" in line:
        line = line.split("//")[0]
    return line


def non_comment_lines(file: TextIO) -> Iterable[str]:
    yield from (rstrip_comment(line)
                for line in file
                if file and not is_comment(line))


def check_name(name1: str, name2: str):
    if name1.lower() != name2.lower():
        raise RuntimeError(
            f"class name doesn't match filename: '{name1}' != '{name2}'")


def handle_weapon_file(path: Path, base_class_name: str) -> Optional[WeaponParseResult]:
    with path.open("r", encoding="latin-1") as file:
        result = WeaponParseResult()
        data = non_comment_lines(file)
        for line in data:
            if not result.class_name:
                match = CLASS_PATTERN.match(line)
                if match:
                    class_name = match.group(1)
                    check_name(class_name, path.stem)
                    parent_name = match.group(2)
                    if not is_weapon_str(parent_name):
                        if class_name == base_class_name:
                            parent_name = base_class_name
                        else:
                            return None
                    result.parent_name = parent_name
                    result.class_name = class_name
                    continue
            if not result.bullet_name:
                match = WEAPON_BULLET_PATTERN.match(line)
                if match:
                    result.bullet_name = match.group(1)
                    continue
            if result.instant_damage:
                match = INSTANT_DAMAGE_PATTERN.match(line)
                if match:
                    result.instant_damage = int(match.group(1))
                    continue
            if result.pre_fire_length:
                match = PRE_FIRE_PATTERN.match(line)
                if match:
                    result.pre_fire_length = int(match.group(1)) // 50
                    continue
            if (result.class_name
                    and result.bullet_name
                    and result.parent_name
                    and result.instant_damage != -1
                    and result.pre_fire_length != -1):
                break
    return result


def handle_bullet_file(path: Path, base_class_name: str) -> Optional[BulletParseResult]:
    with path.open("r", encoding="latin-1") as file:
        result = BulletParseResult()
        data = non_comment_lines(file)
        for line in data:
            if not result.parent_name:
                match = CLASS_PATTERN.match(line)
                if match:
                    class_name = match.group(1)
                    check_name(class_name, path.stem)
                    parent_name = match.group(2)
                    if not is_bullet_str(parent_name):
                        if class_name == base_class_name:
                            parent_name = base_class_name
                        else:
                            return None
                    result.class_name = class_name
                    result.parent_name = parent_name
                    continue
            if result.damage == -1:
                match = DAMAGE_PATTERN.match(line)
                if match:
                    result.damage = int(match.group(1))
                    continue
            if result.speed == math.inf:
                match = SPEED_PATTERN.match(line)
                if match:
                    result.speed = float(match.group(1)) / 50
                    continue
            if not (result.damage_falloff > 0).any():
                match = FALLOFF_PATTERN.match(line)
                if match:
                    result.damage_falloff = parse_interp_curve(
                        match.group(1))
                    continue
            if result.ballistic_coeff == math.inf:
                match = BALLISTIC_COEFF_PATTERN.match(line)
                if match:
                    result.ballistic_coeff = float(match.group(1))
                    continue
            if result.drag_func == DragFunction.Invalid:
                match = DRAG_FUNC_PATTERN.match(line)
                if match:
                    result.drag_func = DragFunction(match.group(1))
                    continue
            if (result.class_name
                    and result.speed != math.inf
                    and result.damage > 0
                    and (result.damage_falloff > 0).any()
                    and result.ballistic_coeff != math.inf
                    and result.drag_func != DragFunction.Invalid):
                break
    return result


def parse_interp_curve(curve: str) -> np.ndarray:
    """The parsed velocity damage falloff curve consists
    of (x,y) pairs, where x is remaining projectile
    speed in m/s and y is the damage scaler at that speed.
    """
    values = []
    curve = re.sub(r"\s", "", curve)
    for match in re.finditer(r"InVal=([\d.]+),OutVal=([\d.]+)", curve):
        # x = math.sqrt(float(match.group(1))) / 50  # UU/s to m/s.
        x = float(match.group(1))
        y = float(match.group(2))
        values.append([x, y])
    return np.array(values)


def is_weapon_str(s: str) -> bool:
    s = s.lower()
    return "roweap_" in s or "weapon" in s


def is_bullet_str(s: str) -> bool:
    s = s.lower()
    return "bullet" in s or "projectile" in s


def process_file(path: Path
                 ) -> Optional[Union[BulletParseResult, WeaponParseResult]]:
    path = path.resolve()
    stem = path.stem
    class_str = ""
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            match = CLASS_PATTERN.match(line)
            if match:
                class_str = match.group(0)
    if is_weapon_str(stem) or is_weapon_str(class_str):
        return handle_weapon_file(path, base_class_name=WEAPON.name)
    elif is_bullet_str(stem) or is_bullet_str(class_str):
        return handle_bullet_file(path, base_class_name=PROJECTILE.name)
    else:
        return None


def resolve_parent(
        obj: ClassBase,
        parse_map: MutableMapping,
        class_map: MutableMapping) -> bool:
    parent_name = parse_map[obj.name].parent_name
    obj.parent = class_map.get(parent_name)
    return obj.parent is not None


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


def gen_delta_time(step: np.longfloat = 0.01
                   ) -> Generator[np.longfloat, None, None]:
    count = 0
    while True:
        if count == 0:
            count += 1
            # Make the first step extra small.
            yield step / np.longfloat(1000)
        else:
            yield step


def main():
    begin = datetime.datetime.now()
    print(f"begin: {begin.isoformat()}")

    bullet_results: MutableMapping[str, BulletParseResult] = CaseInsensitiveDict()
    weapon_results: MutableMapping[str, WeaponParseResult] = CaseInsensitiveDict()

    src_files = [f for f in Path(SRC_DIR).rglob("*.uc")]
    print(f"processing {len(src_files)} .uc files")
    with ThreadPoolExecutor() as executor:
        fs = [executor.submit(process_file, file) for file in src_files]
    result: Optional[ParseResult]
    for future in futures.as_completed(fs):
        result = future.result()
        if result:
            if isinstance(result, WeaponParseResult):
                weapon_results[result.class_name] = result
            elif isinstance(result, BulletParseResult):
                bullet_results[result.class_name] = result

    print(f"found {len(bullet_results)} bullet classes")
    print(f"found {len(weapon_results)} weapon classes")

    # weapon_class = re.sub(
    #     r"roweap_", "", weapon_class, flags=re.IGNORECASE)

    resolved_early = 0
    bullet_classes: MutableMapping[str, Bullet] = CaseInsensitiveDict()
    bullet_classes[PROJECTILE.name] = PROJECTILE
    for class_name, bullet_result in bullet_results.items():
        if class_name == PROJECTILE.name:
            continue
        # May or may not be available, depending on the order.
        parent = bullet_classes.get(bullet_result.parent_name)
        if parent:
            resolved_early += 1
        bullet_classes[class_name] = Bullet(
            name=bullet_result.class_name,
            parent=parent,
            speed=bullet_result.speed,
            damage=bullet_result.damage,
            damage_falloff=bullet_result.damage_falloff,
            drag_func=bullet_result.drag_func,
            ballistic_coeff=bullet_result.ballistic_coeff,
        )

    print(f"{resolved_early} Bullet classes resolved early")
    print(f"{len(bullet_results) - resolved_early} still unresolved")

    to_del = set()
    for class_name, bullet in bullet_classes.items():
        obj = bullet
        while obj.parent != PROJECTILE:
            valid = resolve_parent(
                obj=obj,
                parse_map=bullet_results,
                class_map=bullet_classes,
            )
            if not valid:
                to_del.add(class_name)
                break
            obj = obj.parent

    print(f"discarding {len(to_del)} invalid Bullet classes")
    for td in to_del:
        del bullet_classes[td]

    if not all(b.is_child_of(PROJECTILE) for b in bullet_classes.values()):
        raise RuntimeError("got invalid Bullet classes")

    print(f"{len(bullet_classes)} total Bullet classes")

    resolved_early = 0
    weapon_classes: MutableMapping[str, Weapon] = CaseInsensitiveDict()
    weapon_classes[WEAPON.name] = WEAPON
    for class_name, weapon_result in weapon_results.items():
        if class_name == WEAPON.name:
            continue
        parent = weapon_classes.get(weapon_result.parent_name)
        bullet = bullet_classes.get(weapon_result.bullet_name)
        if parent and bullet:
            resolved_early += 1
        weapon_classes[class_name] = Weapon(
            name=class_name,
            bullet=bullet,
            parent=parent,
            instant_damage=weapon_result.instant_damage,
            pre_fire_length=weapon_result.pre_fire_length,
        )

    print(f"{resolved_early} Weapon classes resolved early")
    print(f"{len(weapon_results) - resolved_early} still unresolved")

    to_del.clear()
    for class_name, weapon in weapon_classes.items():
        obj = weapon
        while obj.parent != WEAPON:
            valid = resolve_parent(
                obj=obj,
                parse_map=weapon_results,
                class_map=weapon_classes,
            )
            if not valid:
                to_del.add(class_name)
                break
            obj = obj.parent

    print(f"discarding {len(to_del)} invalid Weapon classes")
    for td in to_del:
        del weapon_classes[td]

    if not all(w.is_child_of(WEAPON) for w in weapon_classes.values()):
        raise RuntimeError("got invalid Weapon classes")

    if not all(w.get_bullet() for w in weapon_classes.values()):
        for w in weapon_classes.values():
            if not w.get_bullet():
                print(w)
        raise RuntimeError("got Weapon classes without Bullet")

    print(f"{len(weapon_classes)} total Weapon classes")

    bullets_data = []
    for bullet in bullet_classes.values():
        b_data = {
            "name": bullet.name,
            "parent": bullet.parent.name,
            "speed": bullet.get_speed(),
            "damage": bullet.get_damage(),
            "drag_func": bullet.get_drag_func().value,
            "ballistic_coeff": bullet.get_ballistic_coeff(),
        }
        dmg_fo = bullet.get_damage_falloff()
        fo_x, fo_y = interp_dmg_falloff(dmg_fo)
        b_data["damage_falloff"] = dmg_fo.tolist()
        b_data["fo_x"] = fo_x.tolist()
        b_data["fo_y"] = fo_y.tolist()
        bullets_data.append(b_data)
    with open("bullets.json", "w") as f:
        f.write(json.dumps(bullets_data))
    with open("bullets_readable.json", "w") as f:
        f.write(json.dumps(bullets_data, sort_keys=True, indent=4))

    weapons_data = [
        {
            "name": w.name,
            "parent": w.parent.name,
            "bullet": w.get_bullet().name,
            "instant_damage": w.get_instant_damage(),
            "pre_fire_length": w.get_pre_fire_length(),
        }
        for w in weapon_classes.values()
    ]
    with open("weapons.json", "w") as f:
        f.write(json.dumps(weapons_data))
    with open("weapons_readable.json", "w") as f:
        f.write(json.dumps(weapons_data, sort_keys=True, indent=4))

    start_loc = np.array([0.0, 0.0], dtype=np.longfloat) * 50
    # sim = BulletSimulation(
    #     bullet=bullet_classes["akmbullet"],
    #     location=start_loc.copy(),
    # )
    sim = WeaponSimulation(
        weapon=weapon_classes["ROWeap_AK47_AssaultRifle_Type56"],
        location=start_loc.copy(),
        # Initial velocity direction. Magnitude doesn't matter.
        # velocity=np.array([5.0, 0.05]),
        velocity=np.array([1.0, 0.0]),
    )

    np.set_printoptions(suppress=True)
    bullet = sim.bullet
    x, y = interp_dmg_falloff(bullet.get_damage_falloff())
    speed = bullet.get_speed()
    f_ytox = interp1d(y, x, fill_value="extrapolate", kind="linear")
    # f_xtoy = sim.ef_func
    zero_dmg_speed = f_ytox(1.0)
    plt.plot(x, y, marker="o")
    plt.axvline(speed)
    # plt.axvline(zero_dmg_speed)
    # plt.axvline(0)
    plt.text(speed, 0.5, str(speed))
    plt.title(f"{bullet.name} damage falloff curve")
    plt.xlabel(r"bullet speed squared $(\frac{UU}{s})^2$")
    plt.ylabel("energy transfer function")

    l_trajectory_x = []
    l_trajectory_y = []
    l_dmg_curve_x_flight_time = []
    l_dmg_curve_y_damage = []
    l_dmg_curve_x_distance = []
    l_speed_curve_x = []
    l_speed_curve_y = []
    # dt_generator = gen_delta_time(np.longfloat(1) / np.longfloat(500))
    # dt_generator = gen_delta_time(0.0069444444444)
    # prev_dist_incr = 0
    steps = 0
    time_step = 0.016
    sim_time = 2.15
    while sim.flight_time < sim_time:
        steps += 1
        sim.simulate(np.longfloat(time_step))
        flight_time = sim.flight_time
        dmg = sim.calc_damage()
        l_dmg_curve_x_flight_time.append(flight_time)
        l_dmg_curve_y_damage.append(dmg)
        loc = sim.location.copy()
        velocity_ms = np.linalg.norm(sim.velocity) / 50
        l_dmg_curve_x_distance.append(sim.distance_traveled_m)
        l_trajectory_x.append(loc[0])
        l_trajectory_y.append(loc[1])
        l_speed_curve_x.append(flight_time)
        l_speed_curve_y.append(velocity_ms)
        print("flight_time (s) =", flight_time)
        print("damage          =", dmg)
        print("distance (m)    =", sim.distance_traveled_m)
        print("velocity (m/s)  =", velocity_ms)
        print("velocity[1] (Z) =", sim.velocity[1])

    print(f"simulation did {steps} steps")
    print("distance traveled from start to end (Euclidean):",
          np.linalg.norm(sim.location - start_loc) / 50,
          "meters")
    print("bullet distance traveled (simulated accumulation)",
          sim.distance_traveled_m, "meters")

    trajectory_x = np.array(l_trajectory_x) / 50
    trajectory_y = np.array(l_trajectory_y) / 50
    dmg_curve_x_flight_time = np.array(l_dmg_curve_x_flight_time)
    dmg_curve_y_damage = np.array(l_dmg_curve_y_damage)
    dmg_curve_x_distance = np.array(l_dmg_curve_x_distance)
    speed_curve_x = np.array(l_speed_curve_x)
    speed_curve_y = np.array(l_speed_curve_y)

    print("vertical delta (bullet drop) (m):",
          trajectory_y[-1] - trajectory_y[0])

    pref_length = sim.weapon.get_pre_fire_length()
    instant_dmg = sim.weapon.get_instant_damage()

    plt.figure()
    plt.title(f"{bullet.name} trajectory")
    plt.ylabel("y (m)")
    plt.xlabel("x (m)")
    plt.plot(trajectory_x, trajectory_y)

    last_pref = np.where(dmg_curve_y_damage[dmg_curve_x_distance < pref_length])[-1]
    # print(last_pref)
    # dmg_curve_y_damage[last_pref + 1]
    dmg_curve_y_damage_padded = np.insert(dmg_curve_y_damage, last_pref + 1, instant_dmg)
    dmg_curve_x_distance_padded = np.insert(dmg_curve_x_distance, last_pref + 1, pref_length)

    dmg_curve_y_damage[dmg_curve_x_distance <= pref_length] = instant_dmg
    dmg_curve_y_damage_padded[dmg_curve_x_distance_padded <= pref_length] = instant_dmg

    plt.figure()
    plt.title(f"{bullet.name} damage over time")
    plt.xlabel("flight time (s)")
    plt.ylabel("damage")
    plt.plot(dmg_curve_x_flight_time, dmg_curve_y_damage)

    plt.figure()
    plt.title(f"{bullet.name} damage over bullet travel distance")
    plt.xlabel("bullet distance traveled (m)")
    plt.ylabel("damage")
    plt.axvline(pref_length, color="red")
    plt.plot(dmg_curve_x_distance_padded, dmg_curve_y_damage_padded)

    plt.figure()
    plt.title(f"{bullet.name} damage on target at distance")
    plt.xlabel("horizontal distance to target (m)")
    plt.ylabel("damage")
    plt.axvline(pref_length, color="red")
    plt.plot(trajectory_x, dmg_curve_y_damage)

    plt.figure()
    plt.title(f"{bullet.name} speed over time")
    plt.xlabel("flight time (s)")
    plt.ylabel("speed (m/s)")
    plt.plot(speed_curve_x, speed_curve_y)

    end = datetime.datetime.now()
    print(f"end: {end.isoformat()}")
    total_secs = round((end - begin).total_seconds(), 2)
    print(f"processing took {total_secs} seconds")

    plt.show()


if __name__ == "__main__":
    main()
