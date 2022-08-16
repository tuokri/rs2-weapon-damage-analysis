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
from typing import Generator
from typing import MutableMapping
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from requests.structures import CaseInsensitiveDict

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

    print("writing bullets.json")
    with open("bullets.json", "w") as f:
        f.write(json.dumps(bullets_data))
    print("writing bullets_readable.json")
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

    print("writing weapons.json")
    with open("weapons.json", "w") as f:
        f.write(json.dumps(weapons_data))

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
    l_speed_curve_x = []
    l_speed_curve_y = []
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
        l_speed_curve_x.append(flight_time)
        l_speed_curve_y.append(velocity_ms)
        # print("flight_time (s) =", flight_time)
        # print("damage          =", dmg)
        # print("distance (m)    =", sim.distance_traveled_m)
        # print("velocity (m/s)  =", velocity_ms)
        # print("velocity[1] (Z) =", sim.velocity[1])

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
    speed_curve_x = np.insert(speed_curve_x, last_pref, speed_curve_x[last_pref])
    speed_curve_y = np.insert(speed_curve_y, last_pref, speed_curve_y[last_pref])

    trajectory = pd.DataFrame({
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y
    })
    p = (Path(f"./sim_data/") / sim.weapon.name).resolve().with_name("trajectory.csv")
    trajectory.to_csv(p.absolute())


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


def parse_args() -> Namespace:
    ap = ArgumentParser()
    actions = {
        "parse": "parse UnrealScript source files and write classes",
        "simulate": "read class file and run simulations",
    }
    ap.add_argument(
        "action",
        choices=actions.keys(),
        help="the action to perform",
        type=str,
    )
    ap.add_argument(
        "-s", "--src-dir",
        help="Rising Storm 2: Vietnam UnrealScript "
             "source code directory, only used if "
             "'parse' action is chosen",
        type=str,
        default=r"C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\Development",
    )
    ap.add_argument(
        "-p", "--pickle-file",
        help="Weapon classes pickle file the load "
             "for simulations, only used if "
             "'simulate' action is chosen",
        type=str,
        default="weapon_classes.pickle",
    )

    args = ap.parse_args()
    args.action = args.action.lower()
    return args


def main():
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    pickle_file = Path(args.pickle_file).resolve()

    begin = datetime.datetime.now()
    print(f"begin: {begin.isoformat()}")

    if args.action == "parse":
        parse_uscript(src_dir)
    elif args.action == "simulate":
        run_simulations(pickle_file)
    else:
        raise SystemExit("invalid action")

    # with open("weapon_classes.pickle", "rb") as f:
    #     wc2 = preads_weapon_classes(f.read())

    # start_loc = np.array([0.0, 0.0], dtype=np.longfloat)
    # # sim = BulletSimulation(
    # #     bullet=bullet_classes["akmbullet"],
    # #     location=start_loc.copy(),
    # # )
    # sim = WeaponSimulation(
    #     # weapon=weapon_classes["ROWeap_IZH43_Shotgun"],
    #     weapon=weapon_classes["ROWeap_AK47_AssaultRifle"],
    #     location=start_loc.copy(),
    #     # Initial velocity direction. Magnitude doesn't matter.
    #     # velocity=np.array([5.0, 0.05]),
    #     velocity=np.array([1.0, 0.00]),
    # )
    #
    # np.set_printoptions(suppress=True),
    # np.set_printoptions(formatter={"all": lambda _x: str(_x)})
    #
    # # izh = weapon_classes["ROWeap_IZH43_Shotgun"]
    # # pprint(izh)
    # # print(izh.get_bullet().get_speed_uu())
    # # print(izh.get_bullet().get_speed_uu() ** 2)
    # # print(izh.get_bullet().get_speed())
    # # print(izh.get_bullet().get_ballistic_coeff())
    # # print(izh.get_bullet().get_damage_falloff())
    #
    # bullet = sim.bullet
    # x, y = interp_dmg_falloff(bullet.get_damage_falloff())
    # speed = bullet.get_speed()
    # # f_ytox = interp1d(y, x, fill_value="extrapolate", kind="linear")
    # # f_xtoy = sim.ef_func
    # # zero_dmg_speed = f_ytox(1.0)
    # plt.plot(x, y, marker="o")
    # plt.axvline(speed)
    # # plt.axvline(zero_dmg_speed)
    # # plt.axvline(0)
    # plt.text(speed, 0.5, str(speed))
    # plt.title(f"{bullet.name} damage falloff curve")
    # plt.xlabel(r"bullet speed squared $(\frac{UU}{s})^2$")
    # plt.ylabel("energy transfer function")
    #
    # l_trajectory_x = []
    # l_trajectory_y = []
    # l_dmg_curve_x_flight_time = []
    # l_dmg_curve_y_damage = []
    # l_dmg_curve_x_distance = []
    # l_speed_curve_x = []
    # l_speed_curve_y = []
    # # dt_generator = gen_delta_time(np.longfloat(1) / np.longfloat(500))
    # # dt_generator = gen_delta_time(0.0069444444444)
    # # prev_dist_incr = 0
    # steps = 0
    # time_step = 1 / 500  # 0.0165 / 10
    # sim_time = 2.15
    # while sim.flight_time < sim_time:
    #     steps += 1
    #     sim.simulate(time_step)
    #     flight_time = sim.flight_time
    #     dmg = sim.calc_damage()
    #     l_dmg_curve_x_flight_time.append(flight_time)
    #     l_dmg_curve_y_damage.append(dmg)
    #     loc = sim.location.copy()
    #     velocity_ms = np.linalg.norm(sim.velocity) / 50
    #     l_dmg_curve_x_distance.append(sim.distance_traveled_m)
    #     l_trajectory_x.append(loc[0])
    #     l_trajectory_y.append(loc[1])
    #     l_speed_curve_x.append(flight_time)
    #     l_speed_curve_y.append(velocity_ms)
    #     # print("flight_time (s) =", flight_time)
    #     # print("damage          =", dmg)
    #     # print("distance (m)    =", sim.distance_traveled_m)
    #     # print("velocity (m/s)  =", velocity_ms)
    #     # print("velocity[1] (Z) =", sim.velocity[1])
    #
    # print(f"simulation did {steps} steps")
    # print("distance traveled from start to end (Euclidean):",
    #       np.linalg.norm(sim.location - start_loc) / 50,
    #       "meters")
    # print("bullet distance traveled (simulated accumulation)",
    #       sim.distance_traveled_m, "meters")
    #
    # trajectory_x = np.array(l_trajectory_x) / 50
    # trajectory_y = np.array(l_trajectory_y) / 50
    # dmg_curve_x_flight_time = np.array(l_dmg_curve_x_flight_time)
    # dmg_curve_y_damage = np.array(l_dmg_curve_y_damage)
    # dmg_curve_x_distance = np.array(l_dmg_curve_x_distance)
    # speed_curve_x = np.array(l_speed_curve_x)
    # speed_curve_y = np.array(l_speed_curve_y)
    #
    # print("vertical delta (bullet drop) (m):",
    #       trajectory_y[-1] - trajectory_y[0])
    #
    # pref_length = sim.weapon.get_pre_fire_length()
    # instant_dmg = sim.weapon.get_instant_damage()
    #
    # plt.figure()
    # plt.title(f"{bullet.name} trajectory")
    # plt.ylabel("y (m)")
    # plt.xlabel("x (m)")
    # plt.plot(trajectory_x, trajectory_y)
    #
    # last_pref = np.where(dmg_curve_y_damage[dmg_curve_x_distance < pref_length])[-1]
    # # print(last_pref)
    # # dmg_curve_y_damage[last_pref + 1]
    # dmg_curve_y_damage_padded = np.insert(dmg_curve_y_damage, last_pref + 1, instant_dmg)
    # dmg_curve_x_distance_padded = np.insert(dmg_curve_x_distance, last_pref + 1, pref_length)
    #
    # dmg_curve_y_damage[dmg_curve_x_distance <= pref_length] = instant_dmg
    # dmg_curve_y_damage_padded[dmg_curve_x_distance_padded <= pref_length] = instant_dmg
    #
    # plt.figure()
    # plt.title(f"{bullet.name} damage over time")
    # plt.xlabel("flight time (s)")
    # plt.ylabel("damage")
    # plt.plot(dmg_curve_x_flight_time, dmg_curve_y_damage)
    #
    # plt.figure()
    # plt.title(f"{bullet.name} damage over bullet travel distance")
    # plt.xlabel("bullet distance traveled (m)")
    # plt.ylabel("damage")
    # plt.axvline(pref_length, color="red")
    # plt.plot(dmg_curve_x_distance_padded, dmg_curve_y_damage_padded)
    #
    # plt.figure()
    # plt.title(f"{bullet.name} damage on target at distance")
    # plt.xlabel("horizontal distance to target (m)")
    # plt.ylabel("damage")
    # plt.axvline(pref_length, color="red")
    # plt.plot(trajectory_x, dmg_curve_y_damage)
    #
    # plt.figure()
    # plt.title(f"{bullet.name} speed over time")
    # plt.xlabel("flight time (s)")
    # plt.ylabel("speed (m/s)")
    # plt.plot(speed_curve_x, speed_curve_y)

    end = datetime.datetime.now()
    print(f"end: {end.isoformat()}")
    total_secs = round((end - begin).total_seconds(), 2)
    print(f"processing took {total_secs} seconds")

    plt.show()


if __name__ == "__main__":
    main()
