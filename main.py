import configparser
import datetime
import json
import os
from argparse import ArgumentParser
from argparse import Namespace
from collections import deque
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pprint
from typing import Any
from typing import List
from typing import MutableMapping
from typing import Optional

import numpy as np
import pandas as pd
from natsort import natsorted
from orjson import orjson
from requests.structures import CaseInsensitiveDict
from rs2simlib.dataio import pdumps_weapon_classes
from rs2simlib.dataio import ploads_weapon_classes
from rs2simlib.dataio import process_file
from rs2simlib.dataio import resolve_parent
from rs2simlib.fast import sim as fastsim
from rs2simlib.models import Bullet
from rs2simlib.models import BulletParseResult
from rs2simlib.models import DragFunction
from rs2simlib.models import PROJECTILE
from rs2simlib.models import ParseResult
from rs2simlib.models import WEAPON
from rs2simlib.models import Weapon
from rs2simlib.models import WeaponParseResult
from rs2simlib.models import WeaponSimulation
from rs2simlib.models import interp_dmg_falloff
from rs2simlib.models.models import AltAmmoLoadout
from sqlalchemy.dialects.postgresql import insert
from werkzeug.datastructures import MultiDict

import db


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

    print(f"reading '{src_dir.absolute()}'")
    src_files = [f for f in Path(src_dir).rglob("*.uc")]
    print(f"processing {len(src_files)} .uc files")
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
        for b in bullet_classes.values():
            if not b.is_child_of(PROJECTILE):
                print(b.name)
        raise RuntimeError("got invalid Bullet classes")

    print(f"{len(bullet_classes)} total Bullet classes")

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

    # weapon_classes["ROBipodWeapon"].get_bullets()
    # pprint(weapon_classes["ROBipodWeapon"].get_bullets())

    if not all(w.is_child_of(WEAPON) for w in weapon_classes.values()):
        raise RuntimeError("got invalid Weapon classes")

    if not all(any(w.get_bullets()) for w in weapon_classes.values()):
        for w in weapon_classes.values():
            if not all(w.get_bullets()):
                print(w.name, w.get_bullets())
        raise RuntimeError("got Weapon classes without Bullet")

    print(f"{len(weapon_classes)} total Weapon classes")

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

    print("writing bullets.json")
    with open("bullets.json", "w") as f:
        f.write(json.dumps(bullets_data, separators=(",", ":")))

    print("writing bullets_readable.json")
    with open("bullets_readable.json", "w") as f:
        f.write(json.dumps(bullets_data, sort_keys=True, indent=4))

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

    print("writing weapons.json")
    with open("weapons.json", "w") as f:
        f.write(json.dumps(weapons_data, separators=(",", ":")))

    print("writing weapons_readable.json")
    with open("weapons_readable.json", "w") as f:
        f.write(json.dumps(weapons_data, sort_keys=True, indent=4))

    print("writing weapon_classes.pickle")
    with open("weapon_classes.pickle", "wb") as f:
        f.write(pdumps_weapon_classes(weapon_classes))


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
        # print("flight_time (s) =", flight_time)
        # print("damage          =", dmg)
        # print("distance (m)    =", sim.distance_traveled_m)
        # print("velocity (m/s)  =", velocity_ms)
        # print("velocity[1] (Z) =", sim.velocity[1])

    p = (Path(f"./sim_data/") / sim.weapon.name)
    p = p.resolve()
    p.mkdir(parents=True, exist_ok=True)
    p = p / "dummy_name.txt"

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
    speed_curve_x_flight_time = np.array(l_speed_curve_x_flight_time)
    speed_curve_y_velocity_ms = np.array(l_speed_curve_y_velocity_ms)

    # print("vertical delta (bullet drop) (m):",
    #       trajectory_y[-1] - trajectory_y[0])

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
    p = Path(f"./sim_data/") / weapon.name
    p.mkdir(parents=True, exist_ok=True)
    p = p / f"sim_{aim_str}_{bullet.name}.csv"
    p.touch(exist_ok=True)

    df = pd.DataFrame({
        "trajectory_x": results[0],
        "trajectory_y": results[1],
        "damage": results[2],
        "distance": results[3],
        "time_at_flight": results[4],
        "bullet_velocity": results[5],
        "energy_transfer": results[6],
        "power_left": results[7],
    })
    df.to_csv(p.absolute())


def unique_items(seq: List[Any]) -> List[Any]:
    seen = set()
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
        with Path("weapons.json").open("r") as f:
            weapons_metadata = json.load(f)
    except Exception as e:
        print(f"error reading weapons_readable.json: {e}")
        print("make sure script sources are parsed")

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
        weapons_metadata[i]["display_name"] = wm_map.get(
            name)["display_name"] or "NO_DISPLAY_NAME"
        weapons_metadata[i]["short_display_name"] = wm_map.get(
            name)["short_display_name"] or "NO_SHORT_DISPLAY_NAME"

    print("writing weapons.json")
    with Path("weapons.json").open("w") as f:
        f.write(json.dumps(weapons_metadata, separators=(",", ":")))

    print("writing weapons_readable.json")
    with Path("weapons_readable.json").open("w") as f:
        f.write(json.dumps(weapons_metadata, indent=4, sort_keys=True))


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
        print(e)
        print(weapon.name)
        # pprint(weapon)
        raise
    return None


def run_simulations(classes_file: Path):
    """Load weapon classes from pickle file and run
    bullet trajectory, damage, etc. simulations and
    write simulated data to CSV files.
    """
    print(f"reading '{classes_file.absolute()}'")
    weapon_classes = ploads_weapon_classes(classes_file.read_bytes())
    print(f"loaded {len(weapon_classes)} weapons")

    Path("./sim_data/").mkdir(parents=True, exist_ok=True)

    ro_one_shot = weapon_classes["ROOneShotWeapon"]

    # TODO: Pretty dumb way of doing this. Fix later.
    ak47 = weapon_classes["ROWeap_AK47_AssaultRifle"]
    ballistic_proj = ak47.get_bullet(0).find_parent("ROBallisticProjectile")
    sim_classes = []

    pprint(weapon_classes["ROWeap_IZH43_Shotgun"])

    # test = weapon_classes["ROWeap_M79_GrenadeLauncherSmoke_Content"]
    # pprint(test)

    for weapon in weapon_classes.values():
        try:
            # print(weapon.name)
            # print("is roweap  :", weapon.name.lower().startswith("roweap_"))
            # print("is oneshot :", weapon.is_child_of(ro_one_shot))
            # print("has bproj  :", weapon.get_bullet().is_child_of(ballistic_proj))

            if (weapon.name.lower().startswith("roweap_")
                    and not weapon.is_child_of(ro_one_shot)
                    and weapon.get_bullet(0) is not None
                    and weapon.get_bullet(0).is_child_of(ballistic_proj)):
                sim_classes.append(weapon)
        except Exception as e:
            print(e)
            print(weapon.name)
            raise

    # m1 = weapon_classes["ROWeap_M1_Carbine_30rd"]
    # print(m1.get_bullet().name)

    print(f"simulating with {len(sim_classes)} classes")
    pprint([s.name for s in sim_classes])

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        fs = {executor.submit(run_simulation, weapon): weapon
              for weapon in sim_classes}

    done = 0
    for future in futures.as_completed(fs):
        if future.exception():
            # weapon = fs[future]
            print("*" * 80)
            print(f"ERROR!")
            pprint(fs[future])
            print("*" * 80)
            raise future.exception()
        # result = future.result()
        done += 1
        print("done:", done)


def enter_db_data(metadata_dir: Path):
    # TODO: optimize these JSON structures for search.
    weapon_metadata = orjson.loads((metadata_dir / "weapons.json").read_bytes())
    bullet_metadata = orjson.loads((metadata_dir / "bullets.json").read_bytes())

    weapons = ploads_weapon_classes(
        (metadata_dir / "weapon_classes.pickle").read_bytes())

    with db.Session() as session:
        # Special handling for root class.
        wep_root = weapons.pop("Weapon")
        w_meta = next(
            wm for wm in weapon_metadata
            if wm["name"].lower() == wep_root.name.lower())

        insert_stmt = insert(db.Weapon).values(
            name=wep_root.name,
            parent_name=None,
            display_name=w_meta["display_name"],
            short_display_name=w_meta["short_display_name"],
            pre_fire_length=wep_root.get_pre_fire_length() * 50,
        ).on_conflict_do_nothing(
            index_elements=["name"],
        )
        session.execute(insert_stmt)
        session.flush()

        for weapon in weapons.values():
            w_meta = next(
                wm for wm in weapon_metadata
                if wm["name"].lower() == weapon.name.lower())

            parent_name = weapon.parent.name

            w = db.Weapon(
                name=weapon.name,
                parent_name=parent_name,
                display_name=w_meta["display_name"],
                short_display_name=w_meta["short_display_name"],
                pre_fire_length=weapon.get_pre_fire_length() * 50,
            )
            session.add(w)

        session.commit()


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
    args = parse_args()

    begin = datetime.datetime.now()
    print(f"begin: {begin.isoformat()}")

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

    end = datetime.datetime.now()
    print(f"end: {end.isoformat()}")
    total_secs = round((end - begin).total_seconds(), 2)
    print(f"processing took {total_secs} seconds")


if __name__ == "__main__":
    main()
