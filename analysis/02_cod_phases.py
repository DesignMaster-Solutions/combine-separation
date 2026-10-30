"""Combine change-of-direction drills at 10 Hz -> NFL separation.

The stopwatch 3-cone correlates with separation (01_break_retention.py) but only 28 WRs have
a recorded time. Here every 3-cone and short-shuttle rep is split into its turns, and each turn
into phases: deceleration (peak speed -> plant), plant (time spent below half of the entry
speed) and re-acceleration (plant -> 90% of the entry speed). Per player we keep the median turn
and compare it with route-adjusted separation at pass forward (player_play.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/nfl-big-data-bowl-2027")
OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)

DRILLS = ["THREE_CONE_DRILL", "SHORT_SHUTTLE"]
DT = 0.1  # 10 Hz
FEATURES = ["entry_speed", "decel_rate", "plant_time", "reaccel_time", "turn_time"]


def turns(s: np.ndarray, min_entry: float = 3.0) -> list[dict]:
    """Split one rep into turns: each local speed minimum preceded by a run at >= min_entry yd/s."""
    s = pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()
    out = []
    lows = [i for i in range(1, len(s) - 1) if s[i] <= s[i - 1] and s[i] < s[i + 1]]
    prev = 0
    for lo in lows:
        if lo <= prev:
            continue  # still inside the previous turn
        pk = prev + int(s[prev:lo + 1].argmax())
        entry = s[pk]
        if entry < min_entry or s[lo] > 0.6 * entry:
            continue  # stride noise, not a turn
        after = s[lo:]
        back = np.nonzero(after >= 0.9 * entry)[0]
        if len(back) == 0:
            break  # last turn of the rep: no re-acceleration to measure
        rp = lo + int(back[0])
        half = s[pk:rp + 1] < 0.5 * entry
        out.append({
            "entry_speed": entry,
            "decel_rate": (entry - s[lo]) / max((lo - pk) * DT, DT),
            "plant_time": half.sum() * DT,
            "reaccel_time": (rp - lo) * DT,
            "turn_time": (rp - pk) * DT,
            "i_peak": pk, "i_low": lo, "i_back": rp,
        })
        prev = rp
    return out


def self_check() -> None:
    # synthetic rep: 0 -> 6 yd/s, stop to 0.5, back to 6, stop again (no re-accel: dropped)
    s = np.r_[np.linspace(0, 6, 15), np.linspace(6, 0.5, 6), np.linspace(0.5, 6, 11), np.linspace(6, 0, 6)]
    t = turns(s)
    assert len(t) == 1, t
    assert 4.5 < t[0]["entry_speed"] <= 6 and t[0]["reaccel_time"] > 0.5, t


def main() -> None:
    self_check()
    c = pd.read_csv(DATA / "combine_tracking.csv")
    c = c[(c.entity_type == "PLAYER") & c.drill_name.isin(DRILLS)].sort_values("time")
    rows = []
    for (pid, drill, att), g in c.groupby(["nfl_id", "drill_name", "attempt"]):
        for t in turns(g["s"].to_numpy()):
            rows.append({"nfl_id": pid, "drill": drill, **t})
    tr = pd.DataFrame(rows)
    per = tr.groupby("nfl_id")[FEATURES].median().join(tr.groupby("nfl_id").size().rename("n_turns"))

    p = pd.read_csv(DATA / "player_play.csv",
                    usecols=["nfl_id", "route_ran", "separation_at_pass_forward"]).dropna()
    p["sep_over_route"] = p.separation_at_pass_forward - p.groupby("route_ran") \
        .separation_at_pass_forward.transform("mean")
    game = p.groupby("nfl_id").sep_over_route.agg(["mean", "count"]).rename(
        columns={"mean": "sep_over_route", "count": "n_targets"})

    res = pd.read_csv(DATA / "combine_results.csv")[["nfl_id", "combine_position", "three_cone",
                                                     "short_shuttle"]]
    df = per.join(game, how="inner").reset_index().merge(res, on="nfl_id", how="left")
    df = df[(df.n_turns >= 2) & (df.n_targets >= 30)]
    df.to_csv(OUT / "cod_phases_vs_separation.csv", index=False)

    print(f"turns={len(tr)}  players_with_tracking_and_targets={len(df)}  "
          f"with_3cone_time={df.three_cone.notna().sum()}")
    for f in FEATURES + ["three_cone", "short_shuttle"]:
        d = df.dropna(subset=[f])
        r = d[[f, "sep_over_route"]].corr(method="spearman").iloc[0, 1]
        print(f"{f:>14}  n={len(d):>3}  spearman_vs_separation={r:+.3f}")


if __name__ == "__main__":
    main()
