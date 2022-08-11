import datetime
import re
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import TextIO
from typing import Union

import numpy as np

SRC_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\Development"

DAMAGE_PATTERN = re.compile(
    r"^\s*Damage\s*=\s*(\d+).*$", flags=re.IGNORECASE)
FALLOFF_PATTERN = re.compile(
    r"^\s*VelocityDamageFalloffCurve=(\(.*\)).*$", flags=re.IGNORECASE)
CLASS_PATTERN = re.compile(
    r"^\s*class\s+(.*)\s+extends\s+([\w\d_]+)[;\n]+$", flags=re.IGNORECASE)
WEAPON_BULLET_PATTERN = re.compile(
    r"^\s*WeaponProjectiles\(\d+\)\s*=\s*class\s*'(.*)'.*$", flags=re.IGNORECASE)


@dataclass
class ParseResult:
    class_name: str
    parent_name: str


@dataclass
class WeaponParseResult(ParseResult):
    pass


@dataclass
class BulletParseResult(ParseResult):
    pass


@dataclass
class ClassBase:
    name: str
    parent: Optional["ClassBase"]


@dataclass
class Bullet(ClassBase):
    parent: Optional["Bullet"]
    damage: int
    damage_falloff: np.ndarray


@dataclass
class Weapon(ClassBase):
    parent: Optional["Weapon"]
    bullet: Optional[Bullet]


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


def handle_weapon_file(path: Path) -> Optional[Weapon]:
    with path.open("r", encoding="latin-1") as file:
        data = non_comment_lines(file)
        weapon_class = ""
        parent_class = None
        weapon_bullet = None
        for line in data:
            if not weapon_class:
                match = CLASS_PATTERN.match(line)
                if match:
                    weapon_class = match.group(1)
                    parent_name = match.group(2)
                    if "weapon" not in parent_name.lower():
                        if parent_name == "Inventory" and weapon_class == "Weapon":
                            base_weapon = Weapon(
                                name=weapon_class,
                                bullet=None,
                                parent=None)
                            base_weapon.parent = base_weapon
                            return base_weapon
                        else:
                            return None
                    parent_class = Weapon(name=parent_name, bullet=None, parent=None)
                    continue
            if not weapon_bullet:
                match = WEAPON_BULLET_PATTERN.match(line)
                if match:
                    weapon_bullet = BulletBase(name=match.group(1), parent=None)
                    continue
            if weapon_class and weapon_bullet and parent_class:
                return Weapon(
                    name=weapon_class,
                    parent=parent_class,
                    bullet=weapon_bullet)

    if weapon_class and parent_class and not weapon_bullet:
        return Weapon(
            name=weapon_class,
            parent=parent_class,
            bullet=None)

    return None


def handle_bullet_file(path: Path) -> Optional[Bullet]:
    with path.open("r", encoding="latin-1") as file:
        damage = -1
        damage_falloff = None
        data = non_comment_lines(file)
        for line in data:
            if damage == -1:
                match = DAMAGE_PATTERN.match(line)
                if match:
                    damage = int(match.group(1))
                    continue
            if not damage_falloff:
                match = FALLOFF_PATTERN.match(line)
                if match:
                    damage_falloff = np.zeros(1)
                    continue
            if damage > 0 and damage_falloff is not None:
                return Bullet(
                    name=path.stem,
                    damage=damage,
                    damage_falloff=damage_falloff,
                    parent=None,
                )

        if damage > 0 and not damage_falloff:
            return Bullet(
                name=path.stem,
                damage=damage,
                damage_falloff=np.ones(1),
                parent=None,
            )

    return None


def process_file(path: Path) -> Optional[Union[Bullet, Weapon]]:
    path = path.resolve()
    # print(f"processing: '{path}'")
    # print(f"class is: '{path.stem}'")
    class_name = path.stem.lower()
    if "roweap_" in class_name or "weapon" in class_name:
        return handle_weapon_file(path)
    else:
        return handle_bullet_file(path)


def main():
    begin = datetime.datetime.now()
    print(f"start time: {begin.isoformat()}")

    bullets = []
    weapons = []

    src_files = Path(SRC_DIR).rglob("*.uc")
    with ProcessPoolExecutor() as executor:
        fs = [executor.submit(process_file, file) for file in src_files]
        result: Optional[Bullet]
        for future in futures.as_completed(fs):
            result = future.result()
            if result:
                if isinstance(result, Weapon):
                    weapons.append(result)
                elif isinstance(result, Bullet):
                    bullets.append(result)

    print(f"found {len(bullets)} bullet classes")
    print(f"found {len(weapons)} weapon classes")
    for bullet in bullets:
        print(bullet)

    # weapon_class = re.sub(
    #     r"roweap_", "", weapon_class, flags=re.IGNORECASE)

    # Resolve class hierarchy.
    weapon_classes = {
        weapon.name: weapon for weapon in weapons
    }

    base_weapon = weapons.pop(0)
    if base_weapon.name != "Weapon":
        raise RuntimeError(f"'Weapon' should be first element, but got '{base_weapon}'")

    for weapon in weapons:
        parent = weapon_classes[weapon.parent.name]
        weapon.parent = parent
        print(weapon)

    end = datetime.datetime.now()
    print(f"end: {end.isoformat()}")
    total_secs = round((end - begin).total_seconds(), 2)
    print(f"processing took {total_secs} seconds")


if __name__ == "__main__":
    main()
