import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")

    args = ap.parse_args()
    dfs = {}
    for file in args.files:
        path = Path(file).resolve()
        print(f"reading '{path}'")
        dfs[path] = pd.read_csv(path, index_col="time")

    fig = plt.figure()
    ax = fig.add_subplot()
    sim_time = 1 * 751  # Milliseconds.
    df = pd.DataFrame()
    # vel_ax = ax.twinx()
    # vel_ax.set_prop_cycle(cycler(color=["green", "cyan"]))

    pre_fire_len = 25
    weapon2dmg = {
        "ROWeap_AK47_AssaultRifle_Type56": 86,
        "ROWeap_M16A1_AssaultRifle": 95,
    }

    for file, df in dfs.items():
        weapon = file.parent.name
        instant_damage = weapon2dmg[weapon]
        df.loc[df["distance"] <= pre_fire_len, "damage"] = instant_damage
        first_sim_tick = df.loc[df["distance"] >= pre_fire_len].index[0]
        df.loc[first_sim_tick, "damage"] = instant_damage

        df = df.loc[:sim_time]
        bullet = file.stem.split('_')[-1]
        ax = df.plot(
            ax=ax,
            x="distance",
            y="damage",
            xlabel="distance [m]",
            ylabel="damage",
            label=f"{bullet} damage",
        )
        # df.plot(
        #     y="velocity",
        #     ax=vel_ax,
        #     secondary_y=True,
        #     label=f"{bullet} velocity",
        # )

    # vel_ax.right_ax.set_ylabel("velocity [m/s]")

    # def time_to_dmg(x):
    #     return df.loc[df["time"].isin(x)]
    #
    # def dmg_to_time(x):
    #     return df.loc[df["damage"].isin(x)]

    # print(time_col)
    # # extra_ax = ax.secondary_xaxis("top", functions=(dmg_to_time, time_to_dmg))

    extra_ax = ax.twiny()
    dummy_values = np.full(len(df.index), fill_value=df["damage"].mean())
    extra_ax.plot(df.index, dummy_values, linestyle="None")
    extra_ax.set_xlabel("time [ms]")

    ax.axvline(25, color="red", alpha=0.33)
    ax.text(
        26,
        45,
        s="'hitscan' distance (25m) after which damage is based on simulated ballistics",
        color="red"
    )

    # extra_ax.cla()
    # extra_ax.set_xlim(left=0, right=sim_time)
    # ax_xticks = ax.get_xticks()
    # print(ax_xticks)
    # extra_ax.set_xticks(time_col)

    fig.suptitle("Type 56 vs. M16A1 bullet damage simulation")

    plt.show()


if __name__ == "__main__":
    main()
