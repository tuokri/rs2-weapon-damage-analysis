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

import configparser
import itertools
import os
import re
import signal
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from argparse import Namespace
from collections import deque
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat
from typing import Any
from typing import Deque
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Set
from typing import Tuple

import logbook
import numpy as np
import orjson
import pandas as pd
from natsort import natsorted
from requests.structures import CaseInsensitiveDict
from rs2simlib.dataio import pdumps_class_map
from rs2simlib.dataio import ploads_class_map
from rs2simlib.dataio import process_file
from rs2simlib.dataio import resolve_parent
from rs2simlib.fast import sim as fastsim
from rs2simlib.models import Bullet
from rs2simlib.models import BulletParseResult
from rs2simlib.models import ClassLike
from rs2simlib.models import DragFunction
from rs2simlib.models import PROJECTILE
from rs2simlib.models import ParseResult
from rs2simlib.models import WEAPON
from rs2simlib.models import Weapon
from rs2simlib.models import WeaponParseResult
from rs2simlib.models import WeaponSimulation
from rs2simlib.models import interp_dmg_falloff
from rs2simlib.models.models import AltAmmoLoadout
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm import load_only
from werkzeug.datastructures import MultiDict

import rs2simulator
from rs2simulator import db

logger = logbook.Logger("main")

# TODO: make these configurable?
ROOT_DIR = Path(rs2simulator.__path__[0]).parent
SIM_DATA_DIR = (ROOT_DIR / "sim_data").absolute()
BULLETS_JSON = ROOT_DIR / "bullets.json"
BULLETS_READABLE_JSON = ROOT_DIR / "bullets_readable.json"
WEAPONS_JSON = ROOT_DIR / "weapons.json"
WEAPONS_READABLE_JSON = ROOT_DIR / "weapons_readable.json"
WEAPONS_PICKLE = ROOT_DIR / "weapon_classes.pickle"
BULLETS_PICKLE = ROOT_DIR / "bullet_classes.pickle"


def parse_uscript(src_dir: Path):
    """Parse UnrealScript source files and write
    parsed weapon and bullet (projectile) class data
    to JSON and pickle files. The JSON files contain
    flattened representation of the data, whereas the
    pickle files contain the full class hierarchy trees.

    Parsing is done in an ad hoc manner where the
    first pass builds a ParseResult map from
    each UnrealScript file, making its best effort
    to only return ParseResult objects for valid
    Weapon and Projectile (sub)classes. Both ParseResult
    maps are then processed to build class result maps
    that contain ClassBase objects with full inheritance
    chains upto their parent classes (Weapon or Projectile).
    Any invalid objects that do not inherit from either
    valid base class are discarded at this stage.
    """
    bullet_results: MutableMapping[str, BulletParseResult] = CaseInsensitiveDict()
    weapon_results: MutableMapping[str, WeaponParseResult] = CaseInsensitiveDict()

    logger.info("reading '{src_dir}'", src_dir=src_dir.absolute())
    src_files = [f for f in Path(src_dir).rglob("*.uc")]
    logger.info("processing {num} .uc files", num=len(src_files))
    with ThreadPoolExecutor() as executor:
        fs = [executor.submit(process_file, file) for file in src_files]
    result: Optional[ParseResult]
    for future in futures.as_completed(fs):
        result = future.result()
        if result:
            if result.class_name:
                if isinstance(result, WeaponParseResult):
                    weapon_results[result.class_name] = result
                elif isinstance(result, BulletParseResult):
                    bullet_results[result.class_name] = result

    bullet_results = {
        key: bullet_results[key]
        for key in natsorted(bullet_results.keys())
    }

    weapon_results = CaseInsensitiveDict({
        key: weapon_results[key]
        for key in natsorted(weapon_results.keys())
    })

    logger.info("found {num} bullet classes", num=len(bullet_results))
    logger.info("found {num} weapon classes", num=len(weapon_results))

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

    logger.info("{num} bullet classes resolved early",
                num=resolved_early)
    logger.info("{num} still unresolved",
                num=len(bullet_results) - resolved_early)

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

    logger.info("discarding {num} invalid bullet classes", num=len(to_del))
    for td in to_del:
        del bullet_classes[td]

    if not all(b.is_child_of(PROJECTILE) for b in bullet_classes.values()):
        for b in bullet_classes.values():
            if not b.is_child_of(PROJECTILE):
                logger.warn(b.name)
        raise RuntimeError("got invalid bullet classes")

    logger.info("{num} total bullet classes", num=len(bullet_classes))

    resolved_early = 0
    weapon_classes: MutableMapping[str, Weapon] = CaseInsensitiveDict()
    weapon_classes[WEAPON.name] = WEAPON
    for class_name, weapon_result in weapon_results.items():
        if class_name == WEAPON.name:
            continue

        parent = weapon_classes.get(weapon_result.parent_name)

        bullets = [None] * (max(weapon_result.bullet_names, default=0) + 1)
        for idx, bullet_name in weapon_result.bullet_names.items():
            bullets[idx] = bullet_classes.get(bullet_name)

        instant_damages: List[Optional[int]] = [None] * (
                max(weapon_result.instant_damages, default=0) + 1)
        for i_idx, instant_dmg in weapon_result.instant_damages.items():
            instant_damages[i_idx] = instant_dmg

        if parent and any(bullets):
            resolved_early += 1

        alt_ammo_loadouts: List[Optional[AltAmmoLoadout]] = [None] * (
                max(weapon_result.alt_ammo_loadouts, default=0) + 1)
        if weapon_result.alt_ammo_loadouts:
            for idx, alt_lo in weapon_result.alt_ammo_loadouts.items():
                lo_bullets: List[Optional[Bullet]] = [None] * (
                        max(alt_lo.bullet_names, default=0) + 1)
                lo_damages: List[Optional[int]] = [None] * (
                        max(alt_lo.instant_damages, default=0) + 1)
                for d_idx, d in alt_lo.instant_damages.items():
                    lo_damages[d_idx] = d
                for b_idx, b in alt_lo.bullet_names.items():
                    lo_bullets[b_idx] = bullet_classes.get(b)
                alt_ammo_loadouts[idx] = AltAmmoLoadout(
                    name=alt_lo.class_name,
                    parent=None,
                    bullets=lo_bullets,
                    instant_damages=lo_damages,
                )

        weapon_classes[class_name] = Weapon(
            name=class_name,
            bullets=bullets,
            parent=parent,
            instant_damages=instant_damages,
            pre_fire_length=weapon_result.pre_fire_length,
            alt_ammo_loadouts=alt_ammo_loadouts,
        )

    logger.info("{num} weapon classes resolved early",
                num=resolved_early)
    logger.info("{num} still unresolved",
                num=len(weapon_results) - resolved_early)

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

    logger.info("discarding {num} invalid weapon classes", num=len(to_del))
    for td in to_del:
        del weapon_classes[td]

    if not all(w.is_child_of(WEAPON) for w in weapon_classes.values()):
        raise RuntimeError("got invalid weapon classes")

    if not all(any(w.get_bullets()) for w in weapon_classes.values()):
        for w in weapon_classes.values():
            if not all(w.get_bullets()):
                print(w.name, w.get_bullets())
        raise RuntimeError("got weapon classes without bullet")

    logger.info("{num} total weapon classes", num=len(weapon_classes))

    # Second sort may be unnecessary.
    weapon_classes = {
        key: weapon_classes[key]
        for key in natsorted(weapon_classes.keys())
    }
    bullet_classes = {
        key: bullet_classes[key]
        for key in natsorted(bullet_classes.keys())
    }

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
        bullets_data.append({
            key: b_data[key]
            for key in natsorted(b_data.keys())
        })

    logger.info("writing '{f}'", f=BULLETS_JSON)
    BULLETS_JSON.write_bytes(orjson.dumps(bullets_data))

    logger.info("writing '{f}'", f=BULLETS_READABLE_JSON)
    BULLETS_READABLE_JSON.write_bytes(
        orjson.dumps(
            bullets_data,
            option=orjson.OPT_INDENT_2,
        )
    )

    weapons_data = []
    for weapon in weapon_classes.values():
        alt_los = [
            {
                "bullets": [ab.name if ab else None for ab in a.bullets],
                "instant_damages": a.instant_damages,
            } for a in weapon.get_alt_ammo_loadouts()
        ]
        w_data = {
            "name": weapon.name,
            "parent": weapon.parent.name,
            "bullets": [b.name if b else None for b in weapon.get_bullets()],
            "instant_damages": [i or 0 for i in weapon.get_instant_damages()],
            "pre_fire_length": weapon.get_pre_fire_length(),
            "alt_ammo_loadouts": alt_los,
        }
        weapons_data.append({
            key: w_data[key]
            for key in natsorted(w_data.keys())
        })

    logger.info("writing '{f}'", f=WEAPONS_JSON)
    WEAPONS_JSON.write_bytes(orjson.dumps(weapons_data))

    logger.info("writing '{f}'", f=WEAPONS_READABLE_JSON)
    WEAPONS_READABLE_JSON.write_bytes(
        orjson.dumps(
            weapons_data,
            option=orjson.OPT_INDENT_2,
        )
    )

    logger.info("writing '{f}'", f=WEAPONS_PICKLE)
    WEAPONS_PICKLE.write_bytes(pdumps_class_map(weapon_classes))

    logger.info("writing '{f}'", f=BULLETS_PICKLE)
    BULLETS_PICKLE.write_bytes(pdumps_class_map(bullet_classes))


def process_sim(sim: WeaponSimulation, sim_time: float):
    start_loc = sim.location.copy()
    l_trajectory_x = []
    l_trajectory_y = []
    l_dmg_curve_x_flight_time = []
    l_dmg_curve_y_damage = []
    l_dmg_curve_x_distance = []
    l_speed_curve_x_flight_time = []
    l_speed_curve_y_velocity_ms = []
    steps = 0
    time_step = 1 / 500  # 0.0165 / 10
    while sim.flight_time < sim_time:
        steps += 1
        sim.simulate(time_step)
        flight_time = sim.flight_time
        dmg = sim.calc_damage()
        l_dmg_curve_x_flight_time.append(flight_time)
        l_dmg_curve_y_damage.append(dmg)
        loc = sim.location.copy()
        velocity_ms = np.linalg.norm(sim.velocity) / 50
        l_dmg_curve_x_distance.append(sim.distance_traveled_m)
        l_trajectory_x.append(loc[0])
        l_trajectory_y.append(loc[1])
        l_speed_curve_x_flight_time.append(flight_time)
        l_speed_curve_y_velocity_ms.append(velocity_ms)

    p = (SIM_DATA_DIR / sim.weapon.name)
    p = p.resolve()
    p.mkdir(parents=True, exist_ok=True)
    p = p / "dummy_name.txt"

    logger.info("simulation did {steps} steps", steps=steps)
    logger.info(
        "distance traveled from start to end (Euclidean): {x} meters",
        x=np.linalg.norm(sim.location - start_loc) / 50
    )
    logger.info(
        "bullet distance traveled (simulated accumulation) {x} meters",
        x=sim.distance_traveled_m,
    )

    trajectory_x = np.array(l_trajectory_x) / 50
    trajectory_y = np.array(l_trajectory_y) / 50
    dmg_curve_x_flight_time = np.array(l_dmg_curve_x_flight_time)
    dmg_curve_y_damage = np.array(l_dmg_curve_y_damage)
    dmg_curve_x_distance = np.array(l_dmg_curve_x_distance)
    speed_curve_x_flight_time = np.array(l_speed_curve_x_flight_time)
    speed_curve_y_velocity_ms = np.array(l_speed_curve_y_velocity_ms)

    pref_length = sim.weapon.get_pre_fire_length()
    instant_dmg = sim.weapon.get_instant_damage(0)

    last_pref = np.where(dmg_curve_y_damage[dmg_curve_x_distance < pref_length])[-1] + 1
    dmg_curve_y_damage = np.insert(dmg_curve_y_damage, last_pref, instant_dmg)
    dmg_curve_x_distance = np.insert(dmg_curve_x_distance, last_pref, pref_length)

    dmg_curve_y_damage[dmg_curve_x_distance <= pref_length] = instant_dmg

    trajectory_x = np.insert(trajectory_x, last_pref, trajectory_y[last_pref])
    trajectory_y = np.insert(trajectory_y, last_pref, trajectory_y[last_pref])

    dmg_curve_x_flight_time = np.insert(
        dmg_curve_x_flight_time, last_pref, dmg_curve_x_flight_time[last_pref])
    speed_curve_x_flight_time = np.insert(
        speed_curve_x_flight_time, last_pref, speed_curve_x_flight_time[last_pref])
    speed_curve_y_velocity_ms = np.insert(
        speed_curve_y_velocity_ms, last_pref, speed_curve_y_velocity_ms[last_pref])

    damage_curve = pd.DataFrame({
        "dmg_curve_x_flight_time": dmg_curve_x_flight_time,
        "dmg_curve_y_damage": dmg_curve_y_damage,
    })
    p = p.with_name("damage_curve.csv")
    damage_curve.to_csv(p.absolute())

    speed_curve = pd.DataFrame({
        "speed_curve_x_flight_time": speed_curve_x_flight_time,
        "speed_curve_y_velocity_ms": speed_curve_y_velocity_ms,
    })
    p = p.with_name("speed_curve.csv")
    speed_curve.to_csv(p.absolute())

    trajectory = pd.DataFrame({
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
    })
    p = p.with_name("trajectory.csv")
    trajectory.to_csv(p.absolute())


def run_fast_sim(weapon: Weapon, bullet: Bullet,
                 aim_dir: np.ndarray, aim_deg: float,
                 instant_damage: int):
    drag_func: np.int64 = {
        DragFunction.G1: np.int64(1),
        DragFunction.G7: np.int64(7),
    }[bullet.get_drag_func()]

    fo_x, fo_y = interp_dmg_falloff(bullet.get_damage_falloff())

    aim_x: np.float64 = aim_dir[0]
    aim_y: np.float64 = aim_dir[1]

    results = fastsim.simulate(
        sim_time=np.float64(5.0),
        time_step=np.float64(1 / 500),
        drag_func=drag_func,
        ballistic_coeff=np.float64(bullet.get_ballistic_coeff()),
        aim_dir_x=aim_x,
        aim_dir_y=aim_y,
        muzzle_velocity=np.float64(bullet.get_speed_uu()),
        falloff_x=fo_x,
        falloff_y=fo_y,
        bullet_damage=np.int64(bullet.get_damage()),
        instant_damage=np.int64(instant_damage),
        pre_fire_trace_len=np.int64(weapon.get_pre_fire_length() * 50),
        start_loc_x=np.float64(0.0),
        start_loc_y=np.float64(0.0),
    )

    aim_str = f"{aim_deg}_deg".replace(".", "_")
    p = SIM_DATA_DIR / weapon.name
    p.mkdir(parents=True, exist_ok=True)
    p = p / f"sim_{aim_str}_{bullet.name}.csv"
    p.touch(exist_ok=True)

    df = pd.DataFrame({
        # As millisecond integers.
        "time": np.rint(results[4] * 1000).astype(np.uint32),
        "location_x": results[0],
        "location_y": results[1],
        "damage": results[2],
        "distance": results[3],
        "velocity": results[5],
        "energy_transfer": results[6],
        "power_left": results[7],
    })
    df.to_csv(p.absolute(), index=False)


def unique_items(seq: List[Any]) -> List[Any]:
    seen: Set[Any] = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def process_fast_sim(weapon: Weapon, aim_dir: np.ndarray, aim_deg: float):
    bullets = unique_items([b for b in weapon.get_bullets() if b])
    alt_los = [alt for alt in weapon.get_alt_ammo_loadouts() if alt]

    for i, bullet in enumerate(bullets):
        instant_damage = weapon.get_instant_damage(i)
        if bullet:
            run_fast_sim(
                weapon=weapon,
                bullet=bullet,
                aim_dir=aim_dir,
                aim_deg=aim_deg,
                instant_damage=instant_damage,
            )

    for alt_lo in alt_los:
        for i, bullet in enumerate(
                unique_items([b for b in alt_lo.bullets if b])):
            run_fast_sim(
                weapon=weapon,
                bullet=bullet,
                aim_dir=aim_dir,
                aim_deg=aim_deg,
                instant_damage=alt_lo.instant_damages[i],
            )


def parse_localization(path: Path):
    """Read localization file and parse class metadata.
    TODO: should probably also save this to the pickled data.
    """
    try:
        weapons_metadata = orjson.loads(WEAPONS_JSON.read_bytes())
    except Exception as e:
        logger.error("error reading weapons_readable.json: {e}", e=e)
        logger.error("make sure script sources are parsed")
        raise RuntimeError(e)

    cg = configparser.ConfigParser(dict_type=MultiDict, strict=False)
    cg.read(path.absolute())

    retry_keys = []
    for i, wm in enumerate(weapons_metadata):
        name = wm["name"]
        try:
            w_section = cg[name]
            display_name = w_section.get("DisplayName", "").replace('"', "")
            short_name = w_section.get("ShortDisplayName", "").replace('"', "")
            weapons_metadata[i]["display_name"] = display_name
            weapons_metadata[i]["short_display_name"] = short_name
        except KeyError:
            # print("error :", name)
            retry_keys.append(name)

    wm_map = {
        wm["name"]: wm
        for wm in weapons_metadata
    }

    wm_map["Weapon"]["display_name"] = "NO_DISPLAY_NAME"
    wm_map["Weapon"]["short_display_name"] = "NO_SHORT_DISPLAY_NAME"

    retry_queue = deque(natsorted(retry_keys))
    while retry_queue:
        rk = retry_queue.popleft()
        wm = wm_map[rk]

        display_name = wm.get("display_name")
        parent = wm_map.get(wm.get("parent"))
        while not display_name and parent:
            display_name = parent.get("display_name")
            parent = wm_map.get(parent.get("parent"))
            if parent == wm_map.get(parent.get("parent")):
                display_name = parent.get("display_name")
                break
        wm_map[rk]["display_name"] = display_name

        s_display_name = wm.get("short_display_name")
        parent = wm_map.get(wm.get("parent"))
        while not s_display_name and parent:
            s_display_name = parent.get("short_display_name")
            parent = wm_map.get(parent.get("parent"))
            if parent == wm_map.get(parent.get("parent")):
                s_display_name = parent.get("short_display_name")
                break
        wm_map[rk]["short_display_name"] = s_display_name

        if not display_name or not s_display_name:
            retry_queue.append(rk)

    for i, wm in enumerate(weapons_metadata):
        name = wm["name"]
        display_name = "NO_DISPLAY_NAME"
        short_display_name = "NO_SHORT_DISPLAY_NAME"
        wm_map_name = wm_map.get(name)
        if wm_map_name:
            display_name = wm_map_name["display_name"]
            short_display_name = wm_map_name["short_display_name"]
        weapons_metadata[i]["display_name"] = display_name
        weapons_metadata[i]["short_display_name"] = short_display_name

    logger.info("writing '{f}'", f=WEAPONS_JSON)
    WEAPONS_JSON.write_bytes(orjson.dumps(weapons_metadata))

    logger.info("writing '{f}'", f=WEAPONS_READABLE_JSON)
    WEAPONS_READABLE_JSON.write_bytes(
        orjson.dumps(
            weapons_metadata,
            option=orjson.OPT_INDENT_2,
        )
    )


def run_simulation(weapon: Weapon):
    # Degrees up from positive x axis.
    aim_angles = [0, 1, 2, 3, 4, 5]
    aim_rads = np.radians(aim_angles)
    aim_dirs = np.array([(1, np.sin(a)) for a in aim_rads])
    try:
        for aim_dir, aim_deg in zip(aim_dirs, aim_angles):
            # print("aim angle:", np.degrees(np.arctan2(aim_dir[1], aim_dir[0])))
            # sim = WeaponSimulation(
            #     weapon=weapon,
            #     velocity=aim_dir,
            # )
            # process_sim(sim, sim_time=5)
            process_fast_sim(weapon, aim_dir, aim_deg=aim_deg)
    except Exception as e:
        logger.error(e)
        logger.error(weapon.name)
        # pprint(weapon)
        raise
    return None


def run_simulations(classes_file: Path):
    """Load weapon classes from pickle file and run
    bullet trajectory, damage, etc. simulations and
    write simulated data to CSV files.
    """
    logger.info("reading '{f}'", f=classes_file.absolute())
    weapon_classes: MutableMapping[str, Weapon] = ploads_class_map(
        classes_file.read_bytes())
    logger.info("loaded {num} weapons", num=len(weapon_classes))

    SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ro_one_shot = weapon_classes["ROOneShotWeapon"]

    # TODO: Pretty dumb way of doing this. Fix later.
    ak47 = weapon_classes["ROWeap_AK47_AssaultRifle"]
    ballistic_proj = ak47.get_bullet(0).find_parent("ROBallisticProjectile")
    sim_classes = []

    logger.debug(pformat(weapon_classes["ROWeap_IZH43_Shotgun"]))

    for weapon in weapon_classes.values():
        try:
            if (weapon.name.lower().startswith("roweap_")
                    and not weapon.is_child_of(ro_one_shot)
                    and weapon.get_bullet(0) is not None
                    and weapon.get_bullet(0).is_child_of(ballistic_proj)):
                sim_classes.append(weapon)
        except Exception as e:
            logger.error(e)
            logger.error(weapon.name)
            raise

    logger.info("simulating with {num} classes", num=len(sim_classes))
    logger.info(pformat([s.name for s in sim_classes]))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        fs = {executor.submit(run_simulation, weapon): weapon
              for weapon in sim_classes}

    done = 0
    for future in futures.as_completed(fs):
        if future.exception():
            logger.error("*" * 80)
            logger.error(f"ERROR!")
            logger.error(pformat(fs[future]))
            logger.error("*" * 80)
            raise RuntimeError(future.exception())
        done += 1
        logger.info("done:", done)


def get_immediate_children(
        parent: ClassLike,
        classes: Iterable[ClassLike]
) -> Iterable[ClassLike]:
    return [
        c for c in classes
        if c != parent
        if c.parent.name == parent.name
    ]


def hierarchy_list(
        root: ClassLike,
        classes: Iterable[ClassLike],
) -> List[ClassLike]:
    # return [
    #     root,
    #     *[c for children in [hierarchy_list(child, classes)
    #                          for child in get_immediate_children(root, classes)]
    #       for c in children]
    # ]

    ret = [root]

    children = get_immediate_children(root, classes)
    for child in children:
        ret.extend(hierarchy_list(child, classes))

    return ret


def ordered_class_likes(
        root_name: str,
        class_map: MutableMapping[str, ClassLike],
) -> List[ClassLike]:
    c_root = class_map[root_name]
    return hierarchy_list(c_root, class_map.values())


def insert_weapons(
        weapon_metadata: dict,
        weapon_classes: List[Weapon],
):
    with db.Session(autoflush=False) as session, session.begin():
        w: Optional[db.models.Weapon] = None
        parent_id = None
        loadouts = []

        for weapon in weapon_classes:
            w_meta = next(
                wm for wm in weapon_metadata
                if wm["name"].lower() == weapon.name.lower())

            if w is not None:
                parent_id = session.scalar(
                    select(db.models.Weapon.id).where(
                        db.models.Weapon.name == weapon.parent.name
                    )
                )

            w = db.models.Weapon(
                name=weapon.name,
                display_name=w_meta["display_name"],
                short_display_name=w_meta["short_display_name"],
                parent_id=parent_id,
                pre_fire_length=weapon.get_pre_fire_length() * 50,
            )

            session.add(w)
            session.flush()
            session.refresh(w)

            b_d_values = [
                (b, d) for b, d in zip(
                    weapon.get_bullets(), weapon.get_instant_damages()
                ) if b
            ]
            loadouts.extend(make_loadouts(session, w, b_d_values))

            alt_los = [lo for lo in weapon.get_alt_ammo_loadouts() if lo]
            alt_lo_bullets = itertools.chain.from_iterable(
                [lo.bullets for lo in alt_los])
            alt_lo_damages = itertools.chain.from_iterable(
                [lo.instant_damages for lo in alt_los])
            lo_b_d_values = [
                (b, d) for b, d in zip(
                    alt_lo_bullets,
                    alt_lo_damages,
                ) if b
            ]
            loadouts.extend(make_loadouts(session, w, lo_b_d_values))

        session.add_all(loadouts)


def make_loadouts(
        session: Session,
        weapon: db.models.Weapon,
        b_d_values: Iterable[Tuple[Bullet, int]]
) -> List[db.models.AmmoLoadout]:
    loadouts = []
    for bullet, instant_damage in b_d_values:
        loadouts.append(db.models.AmmoLoadout(
            weapon_id=weapon.id,
            bullet_id=session.scalar(
                select(db.models.Bullet.id).where(
                    db.models.Bullet.name == bullet.name
                ),
            ),
            instant_damage=instant_damage,
        ))
    return loadouts


def insert_bullets(bullet_classes: List[Bullet]):
    with db.Session(autoflush=False) as session, session.begin():
        b: Optional[db.models.Bullet] = None
        parent_id = None

        for bullet in bullet_classes:
            if b is not None:
                parent_id = session.scalar(
                    select(db.models.Bullet.id).where(
                        db.models.Bullet.name == bullet.parent.name
                    )
                )

            b = db.models.Bullet(
                name=bullet.name,
                parent_id=parent_id,
                ballistic_coeff=bullet.get_ballistic_coeff(),
                damage=bullet.get_damage(),
                drag_func=bullet.get_drag_func_int(),
                speed=bullet.get_speed_uu(),
            )

            session.add(b)
            session.flush()
            session.refresh(b)

            dmg_fo = interp_dmg_falloff(bullet.get_damage_falloff())

            for i, (x, y) in enumerate(zip(*dmg_fo)):
                dmg_falloff = db.models.DamageFalloff(
                    index=i,
                    x=x,
                    y=y,
                    bullet_id=b.id,
                )
                session.add(dmg_falloff)


def enter_db_data(metadata_dir: Path):
    db.drop_create_all()

    # TODO: optimize these JSON structures for search.
    weapon_metadata = orjson.loads(
        (metadata_dir / "weapons.json").read_bytes())

    weapons: MutableMapping[str, Weapon] = ploads_class_map(
        (metadata_dir / "weapon_classes.pickle").read_bytes())

    bullets: MutableMapping[str, Bullet] = ploads_class_map(
        (metadata_dir / "bullet_classes.pickle").read_bytes())

    ordered_weapons: List[Weapon] = ordered_class_likes(
        root_name="Weapon",
        class_map=weapons,
    )
    ordered_bullets: List[Bullet] = ordered_class_likes(
        root_name="Projectile",
        class_map=bullets,
    )

    # TODO: better filtering?
    ordered_weapons = [
        w for w in ordered_weapons
        if "roweap" in w.name.lower()
        if "_content" not in w.name.lower()
        if (
                "shotgun" in w.name.lower()
                or
                "rifle" in w.name.lower()
                or
                "pistol" in w.name.lower()
                or
                "bar" in w.name.lower()
                or
                "lmg" in w.name.lower()
                or
                "carbine" in w.name.lower()
                or
                "gpmg" in w.name.lower()
                or
                "smg" in w.name.lower()
                or
                "hmg" in w.name.lower()
        )
    ]

    # TODO: could just return bullet and weapon IDs here,
    #   instead of doing a new select right after for
    #   simulation data insertion.
    insert_bullets(ordered_bullets)
    insert_weapons(weapon_metadata, ordered_weapons)

    with db.Session() as session:
        db_weapons = session.scalars(
            select(db.models.Weapon).options(load_only(
                db.models.Weapon.name,
                db.models.Weapon.id,
            ))
        ).all()
        db_bullets = session.scalars(
            select(db.models.Bullet).options(load_only(
                db.models.Bullet.name,
                db.models.Bullet.id,
            ))
        ).all()

    weapon_ids = {weapon.name: weapon.id for weapon in db_weapons}
    bullet_ids = {bullet.name: bullet.id for bullet in db_bullets}

    db.engine().dispose(close=False)
    fs = {}
    try:
        with ThreadPoolExecutor(
                max_workers=os.cpu_count(),
        ) as executor:
            data_dirs = [
                p.absolute()
                for p in SIM_DATA_DIR.iterdir()
                if p.is_dir()
            ]
            for data_dir in data_dirs:
                # TODO: better filtering.
                if data_dir.name in [w.name for w in ordered_weapons]:
                    fs[executor.submit(
                        process_sim_csv,
                        data_dir=data_dir,
                        weapon_ids=weapon_ids,
                        bullet_ids=bullet_ids,
                    )] = data_dir

        for f in futures.as_completed(fs):
            exc = f.exception()
            if exc:
                raise exc
    except Exception as e:
        STOP_EVENT.set()
        logger.error("{e_type}: {e}", e_type=type(e).__name__, e=e)
        logger.exception(e)
        for p in PROCESSES:
            p.send_signal(signal.CTRL_BREAK_EVENT)
        for f in fs:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


PROCESS_CSV_SCRIPT = str(
    Path("rs2simulator/scripts/enter_sim_data.py").absolute()
)
PROCESSES: Deque[subprocess.Popen] = deque()
STOP_EVENT = threading.Event()


def process_sim_csv(data_dir: Path, bullet_ids: dict, weapon_ids: dict):
    if STOP_EVENT.is_set():
        logger.warn("stop event is set, aborting")
        return

    bullet_names = set()

    # TODO: this is currently calculated twice. Pass the result of this
    #  calculation to the subprocess to avoid doing it twice.
    sim_file_pat = re.compile(r"sim_(\d)_deg_(.+)\.csv")
    csv_paths = list(data_dir.glob("*.csv"))
    for csv_path in csv_paths:
        match = sim_file_pat.match(csv_path.name)
        if match:
            bullet_names.add(match.group(2))

    sub_weapon_ids = {data_dir.name: weapon_ids[data_dir.name]}
    sub_bullet_ids = {bullet_name: bullet_ids[bullet_name]
                      for bullet_name in bullet_names}

    weapon_id_args = [f"{name}={ident}"
                      for name, ident in sub_weapon_ids.items()]
    bullet_id_args = [f"{name}={ident}"
                      for name, ident in sub_bullet_ids.items()]

    p_args = [
        sys.executable,
        PROCESS_CSV_SCRIPT,
        str(data_dir),
        "--weapon-ids",
        *weapon_id_args,
        "--bullet-ids",
        *bullet_id_args,
    ]
    p = subprocess.Popen(
        args=p_args,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    PROCESSES.append(p)

    while p.poll() is None:
        if STOP_EVENT.is_set():
            logger.warn("stop event is set, terminating")
            p.terminate()
            break
        time.sleep(0.1)

    r = p.wait()
    if r != 0:
        STOP_EVENT.set()
        raise RuntimeError(f"process failed with exit code: {r}")


src_default = (
    r"C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\Development")
loc_default = (
    r"C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\ROGame\Localization\INT\ROGame.int")


def parse_args() -> Namespace:
    ap = ArgumentParser()
    group = ap.add_argument_group("mutually exclusive arguments")
    group.add_argument(
        "-p", "--parse-src",
        help="parse Rising Storm 2: Vietnam UnrealScript source code",
        default=None,
        const=src_default,
        nargs="?",
        metavar="USCRIPT_SRC_DIR",
        action="store",
    )
    group.add_argument(
        "-l", "--parse-localization",
        help="parse localization file and read weapon metadata, "
             "source code should be parsed before parsing localization",
        default=None,
        const=loc_default,
        nargs="?",
        metavar="LOCALIZATION_FILE",
        action="store",
    )
    group.add_argument(
        "-s", "--simulate",
        help="load weapon class pickle file and run simulations, "
             "both source code and localization should be "
             "parsed before running simulations",
        default=None,
        const="weapon_classes.pickle",
        nargs="?",
        metavar="CLASS_PICKLE_FILE",
        action="store",
    )
    group.add_argument(
        "-e", "--enter-db-data",
        help="TODO: help",
        default=None,
        const=".",
        nargs="?",
        metavar="METADATA_DIR",
        action="store",
    )

    args = ap.parse_args()
    mutex_args = (args.simulate, args.parse_src,
                  args.parse_localization, args.enter_db_data)
    if mutex_args.count(None) != len(mutex_args) - 1:
        ap.error("exactly one mutually exclusive argument is required")

    return args


def main():
    def handler(*_):
        STOP_EVENT.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    args = parse_args()

    begin = time.perf_counter()
    logger.info(f"begin: {begin}")

    if args.parse_src:
        parse_uscript(Path(args.parse_src).resolve())
    elif args.parse_localization:
        parse_localization(Path(args.parse_localization).resolve())
    elif args.simulate:
        run_simulations(Path(args.simulate).resolve())
    elif args.enter_db_data:
        enter_db_data(Path(args.enter_db_data).resolve())
    else:
        raise SystemExit("no valid action specified")

    end = time.perf_counter()
    logger.info("end: {end}", end=end)
    total_secs = end - begin
    logger.info("processing took {total} seconds", total=total_secs)


if __name__ == "__main__":
    setup = logbook.NestedSetup([
        logbook.NullHandler(),
        logbook.RotatingFileHandler("main.log"),
        logbook.StreamHandler(sys.stdout, bubble=True),
    ])
    with setup.applicationbound():
        main()

    # import timeit
    #
    # print(timeit.timeit(
    #     "enter_db_data(Path())",
    #     setup="from __main__ import enter_db_data; from pathlib import Path",
    #     number=100,
    # ))
