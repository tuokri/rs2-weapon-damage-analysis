import datetime
import re
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable
from typing import MutableMapping
from typing import Optional
from typing import TextIO
from typing import Union

import numpy as np
from requests.structures import CaseInsensitiveDict

SRC_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\Development"

DAMAGE_PATTERN = re.compile(
    r"^\s*Damage\s*=\s*(\d+).*$", flags=re.IGNORECASE)
FALLOFF_PATTERN = re.compile(
    r"^\s*VelocityDamageFalloffCurve\s*=\s*\(Points\s*=\s*(\(.*\))\).*$", flags=re.IGNORECASE)
CLASS_PATTERN = re.compile(
    r"^\s*class\s+([\w]+)\s+extends\s+([\w\d_]+)\s*.*[;\n\s]+$", flags=re.IGNORECASE)
WEAPON_BULLET_PATTERN = re.compile(
    r"^\s*WeaponProjectiles\(\d+\)\s*=\s*class\s*'(.*)'.*$", flags=re.IGNORECASE)


@dataclass
class ParseResult:
    class_name: str = ""
    parent_name: str = ""


@dataclass
class WeaponParseResult(ParseResult):
    bullet_name: str = ""


@dataclass
class BulletParseResult(ParseResult):
    damage: int = -1
    damage_falloff: np.ndarray = np.array([0, 0])


@dataclass
class ClassBase:
    name: str = field(hash=True)
    parent: Optional["ClassBase"]

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
        # while parent:
        #     next_parent = parent.parent
        #     if next_parent.name == obj.name:
        #         return True
        #     # Found base class.
        #     elif next_parent.name == parent.name:
        #         return False
        # return False


@dataclass
class Bullet(ClassBase):
    parent: Optional["Bullet"]
    damage: int
    damage_falloff: np.ndarray


@dataclass
class Weapon(ClassBase):
    parent: Optional["Weapon"]
    bullet: Optional[Bullet]


PROJECTILE = Bullet(
    name="Projectile",
    damage=0,
    damage_falloff=np.array([0, 0]),
    parent=None,
)
PROJECTILE.parent = PROJECTILE
WEAPON = Weapon(
    name="Weapon",
    bullet=PROJECTILE,
    parent=None,
)
WEAPON.parent = WEAPON


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
            if not result.damage_falloff.all():
                match = FALLOFF_PATTERN.match(line)
                if match:
                    result.damage_falloff = parse_interp_curve(
                        match.group(1))
                    continue
            if (result.class_name
                    and result.damage > 0
                    and result.damage_falloff.all()):
                break
    return result


def parse_interp_curve(curve: str) -> np.ndarray:
    # (Points=((InVal=95062500,OutVal=1.0),(InVal=380250000,OutVal=0.55)))
    points = curve.split["Points="]
    return np.array([1, 1])


def is_weapon_str(s: str) -> bool:
    s = s.lower()
    return "roweap_" in s or "weapon" in s


def is_bullet_str(s: str) -> bool:
    s = s.lower()
    return "bullet" in s or "projectile" in s


def process_file(path: Path
                 ) -> Optional[Union[BulletParseResult, WeaponParseResult]]:
    path = path.resolve()
    # print(f"processing: '{path}'")
    # print(f"class is: '{path.stem}'")
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


def main():
    begin = datetime.datetime.now()
    print(f"begin: {begin.isoformat()}")

    bullet_results: MutableMapping[str, BulletParseResult] = CaseInsensitiveDict()
    weapon_results: MutableMapping[str, WeaponParseResult] = CaseInsensitiveDict()

    src_files = Path(SRC_DIR).rglob("*.uc")
    with ProcessPoolExecutor() as executor:
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
        # May or may not be available, depending on the order.
        parent = bullet_classes.get(bullet_result.parent_name)
        if parent:
            resolved_early += 1
        bullet_classes[class_name] = Bullet(
            name=bullet_result.class_name,
            parent=parent,
            damage=bullet_result.damage,
            damage_falloff=bullet_result.damage_falloff,
        )

    print(f"{resolved_early} bullet classes resolved early")
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

    for td in to_del:
        del bullet_classes[td]

    if not all(b.is_child_of(PROJECTILE) for b in bullet_classes.values()):
        raise SystemExit("got invalid Bullet classes")

    print(f"{len(bullet_classes)} total bullet classes")

    resolved_early = 0
    weapon_classes: MutableMapping[str, Weapon] = CaseInsensitiveDict()
    weapon_classes[WEAPON.name] = WEAPON
    for class_name, weapon_result in weapon_results.items():
        parent = weapon_classes.get(weapon_result.parent_name)
        if parent:
            resolved_early += 1
        weapon_classes[class_name] = Weapon(
            name=class_name,
            bullet=bullet_classes.get(weapon_result.bullet_name),
            parent=parent,
        )

    print(f"{resolved_early} weapon classes resolved early")
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

    for td in to_del:
        del weapon_classes[td]

    if not all(w.is_child_of(WEAPON) for w in weapon_classes.values()):
        raise SystemExit("got invalid Weapon classes")

    print(f"{len(weapon_classes)} total weapon classes")

    end = datetime.datetime.now()
    print(f"end: {end.isoformat()}")
    total_secs = round((end - begin).total_seconds(), 2)
    print(f"processing took {total_secs} seconds")


if __name__ == "__main__":
    main()
