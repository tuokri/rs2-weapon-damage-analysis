import configparser
import datetime
import json
import os
from argparse import ArgumentParser
from argparse import Namespace
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pprint
from typing import Dict
from typing import Generator
from typing import MutableMapping
from typing import Optional

import numpy as np
import pandas as pd
from natsort import natsorted
from requests.structures import CaseInsensitiveDict
from werkzeug.datastructures import MultiDict

from dataio import pdumps_weapon_classes
from dataio import ploads_weapon_classes
from dataio import process_file
from dataio import resolve_parent
from models import Bullet
from models import BulletParseResult
from models import PROJECTILE
from models import ParseResult
from models import WEAPON
from models import Weapon
from models import WeaponParseResult
from models import WeaponSimulation
from models import interp_dmg_falloff


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


def parse_uscript(src_dir: Path):
    """Parse UnrealScript source files and write
    parsed weapon classes to JSON and pickle files.
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
        w_data = {
            "name": weapon.name,
            "parent": weapon.parent.name,
            "bullet": weapon.get_bullet().name,
            "instant_damage": weapon.get_instant_damage(),
            "pre_fire_length": weapon.get_pre_fire_length(),
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
    instant_dmg = sim.weapon.get_instant_damage()

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


def parse_localization(path: Path):
    """Read localization file and parse class metadata."""
    try:
        with Path("weapons.json").open("r") as f:
            weapons_metadata = json.load(f)
    except Exception as e:
        print(f"error reading weapons_readable.json: {e}")
        print("make sure script sources are parsed")

    cg = configparser.ConfigParser(dict_type=MultiDict, strict=False)
    cg.read(path.absolute())

    retry_keys = set()
    for i, wm in enumerate(weapons_metadata):
        name = wm["name"]
        try:
            w_section = cg[name]
            display_name = w_section["DisplayName"].replace('"', "")
            short_name = w_section["ShortDisplayName"].replace('"', "")
            weapons_metadata[i]["display_name"] = display_name
            weapons_metadata[i]["short_display_name"] = short_name
        except KeyError:
            retry_keys.add(name)

    wm_map: Dict[str, Dict] = {}
    if retry_keys:
        for wm in weapons_metadata:
            wm_map[wm["name"]] = wm

    for rk in retry_keys:
        wm = wm_map[rk]

        display_name = wm.get("display_name")
        parent = wm_map.get(wm.get("parent"))
        while not display_name and parent:
            display_name = parent.get("display_name")
            parent = wm_map.get(wm.get("parent"))
            if parent == wm_map.get(wm.get("parent")):
                break
        if not display_name:
            display_name = "NO_DISPLAY_NAME"
        wm_map[rk]["display_name"] = display_name

        s_display_name = wm.get("short_display_name")
        parent = wm_map.get(wm.get("parent"))
        while not s_display_name and parent:
            s_display_name = parent.get("short_display_name")
            parent = wm_map.get(wm.get("parent"))
            if parent == wm_map.get(wm.get("parent")):
                break
        if not s_display_name:
            s_display_name = "NO_SHORT_DISPLAY_NAME"
        wm_map[rk]["short_display_name"] = s_display_name

    weapons_metadata = list(wm_map.values())

    print("writing weapons.json")
    with Path("weapons.json").open("w") as f:
        f.write(json.dumps(weapons_metadata, separators=(",", ":")))

    print("writing weapons_readable.json")
    with Path("weapons_readable.json").open("w") as f:
        f.write(json.dumps(weapons_metadata, indent=4, sort_keys=True))


def run_simulation(weapon: Weapon):
    # Degrees up from positive x axis.
    aim_angles = np.radians([0, 1, 2, 3, 4, 5])
    aim_dirs = np.array([(1, np.sin(a)) for a in aim_angles])
    try:
        for aim_dir in aim_dirs:
            # print("aim angle:", np.degrees(np.arctan2(aim_dir[1], aim_dir[0])))
            sim = WeaponSimulation(
                weapon=weapon,
                velocity=aim_dir,
            )
            process_sim(sim, sim_time=5)
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
    with classes_file.open("rb") as f:
        weapon_classes = ploads_weapon_classes(f.read())
    print(f"loaded {len(weapon_classes)} weapons")

    Path("./sim_data/").mkdir(parents=True, exist_ok=True)

    ro_one_shot = weapon_classes["ROOneShotWeapon"]

    # TODO: Pretty dumb way of doing this. Fix later.
    ak47 = weapon_classes["ROWeap_AK47_AssaultRifle"]
    ballistic_proj = ak47.get_bullet().find_parent("ROBallisticProjectile")
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
                    and weapon.get_bullet().is_child_of(ballistic_proj)):
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
        fs = [executor.submit(run_simulation, weapon)
              for weapon in sim_classes]

    done = 0
    for future in futures.as_completed(fs):
        if future.exception():
            raise future.exception()
        result = future.result()
        done += 1
        print("done:", done)


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

    args = ap.parse_args()
    mutex_args = (args.simulate, args.parse_src, args.parse_localization)
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
    else:
        raise SystemExit("no valid action specified")

    end = datetime.datetime.now()
    print(f"end: {end.isoformat()}")
    total_secs = round((end - begin).total_seconds(), 2)
    print(f"processing took {total_secs} seconds")


if __name__ == "__main__":
    main()
