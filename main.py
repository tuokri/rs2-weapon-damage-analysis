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

    def __hash__(self) -> int:
        return super().__hash__()

    def get_bullet(self) -> Bullet:
        return self.get_attr("bullet")


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
)
WEAPON.parent = WEAPON

str_to_df = {
    DragFunction.G1: drag_g1,
    DragFunction.G7: drag_g7,
}


@dataclass
class BulletSimulation:
    bullet: Bullet
    flight_time: float = 0
    bc_inverse: float = 0
    velocity: np.ndarray = np.array([1, 0], dtype=np.float64)
    location: np.ndarray = np.array([0, 1], dtype=np.float64)
    ef_func: Callable = field(init=False)

    def __post_init__(self):
        self.bc_inverse = self.bullet.get_ballistic_coeff() / 1
        # Initial velocity unit vector * muzzle speed.
        v_normalized = self.velocity / np.linalg.norm(self.velocity)
        self.velocity = v_normalized * self.bullet.get_speed_uu()
        fo_x, fo_y = interp_dmg_falloff(self.bullet.get_damage_falloff())
        self.ef_func = interp1d(
            fo_x, fo_y, kind="linear", fill_value="extrapolate")

    def calc_drag_coeff(self, mach: float) -> float:
        return str_to_df[self.bullet.get_drag_func()](mach)

    def simulate(self, delta_time: float):
        if delta_time <= 0:
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
                * SCALE_FACTOR * (-1 * (self.velocity / v_size)))
        # FLOAT ZAcceleration = 490.3325 * DeltaTime; // 490.3325 UU/S = 9.806 65 M/S
        self.velocity[1] -= (490.3325 * delta_time)
        self.location += (self.velocity * delta_time)

    def calc_damage(self) -> float:
        v_size = np.linalg.norm(self.velocity) / 50  # m/s.
        print("speed =", v_size, "m/s")
        power_left = v_size / self.bullet.get_speed()
        print("power_left =", power_left * 100, "%")
        damage = self.bullet.get_damage() * power_left
        energy_transfer = self.ef_func(v_size)
        print("energy_transfer=", energy_transfer)
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
            if (result.class_name
                    and result.bullet_name
                    and result.parent_name):
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
        x = math.sqrt(float(match.group(1))) / 50  # UU/s to m/s.
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
    if is_weapon_str(stem):
        return handle_weapon_file(path, base_class_name=WEAPON.name)
    elif is_bullet_str(stem):
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
    try:
        f_ytox = interp1d(y, x, fill_value="extrapolate", kind="linear")
    except ValueError:
        return x, y
    zero_dmg_speed = f_ytox(1)
    x = np.insert(x, 0, zero_dmg_speed)
    y = np.insert(y, 0, 1)
    return x, y


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
        }
        for w in weapon_classes.values()
    ]
    with open("weapons.json", "w") as f:
        f.write(json.dumps(weapons_data))
    with open("weapons_readable.json", "w") as f:
        f.write(json.dumps(weapons_data, sort_keys=True, indent=4))

    np.set_printoptions(suppress=True)
    bullet = bullet_classes["AKMBullet"]
    x, y = interp_dmg_falloff(bullet.get_damage_falloff())
    speed = bullet.get_speed()
    f_ytox = interp1d(y, x, fill_value="extrapolate", kind="linear")
    f_xtoy = interp1d(x, y, fill_value="extrapolate", kind="linear")
    zero_dmg_speed = f_ytox(1)
    plt.plot(x, y, marker="o")
    plt.axvline(speed)
    plt.axvline(zero_dmg_speed)
    plt.axvline(0)
    plt.text(speed, 0.5, str(speed))
    plt.title(f"{bullet.name} damage falloff curve")

    start_loc = np.array([0.0, 100.0], dtype=np.float64)
    sim = BulletSimulation(
        bullet=bullet_classes["AKMBullet"],
        location=start_loc.copy(),
    )
    delta_time = 0.5
    trajectory_x = []
    trajectory_y = []
    dmg_curve_x = []
    dmg_curve_y = []
    speed_curve_x = []
    speed_curve_y = []
    while sim.flight_time < 10:
        sim.simulate(delta_time)
        flight_time = sim.flight_time
        print("flight_time =", flight_time)
        dmg = sim.calc_damage()
        print("damage =", dmg)
        dmg_curve_x.append(flight_time)
        dmg_curve_y.append(dmg)
        loc = sim.location / 50
        trajectory_x.append(loc[0])
        trajectory_y.append(loc[1])
        speed_curve_x.append(flight_time)
        speed_curve_y.append(np.linalg.norm(sim.velocity) / 50)

    print("distance traveled: ", np.linalg.norm(sim.location - start_loc) / 50, " meters")

    plt.figure()
    plt.title(f"{bullet.name} trajectory")
    plt.plot(trajectory_x, trajectory_y)

    plt.figure()
    plt.title(f"{bullet.name} damage over time")
    plt.plot(dmg_curve_x, dmg_curve_y)

    plt.figure()
    plt.title(f"{bullet.name} speed over time")
    plt.plot(speed_curve_x, speed_curve_y)

    plt.show()

    end = datetime.datetime.now()
    print(f"end: {end.isoformat()}")
    total_secs = round((end - begin).total_seconds(), 2)
    print(f"processing took {total_secs} seconds")


if __name__ == "__main__":
    main()
