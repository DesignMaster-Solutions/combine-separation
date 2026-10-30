"""Measure the route break itself in NFL game tracking (10 Hz, 2023 season) and link it to the Combine.

04_combined.py found that the speed a WR carries into the Combine reversals (+0.32 composite with the
bend ratio) tracks route-adjusted NFL separation. That is still drill-to-outcome. Here the same kind
of movement is measured on the field: for each targeted-or-not WR route with a break (out, in, corner,
post, slant, hitch), between the snap and the throw, find the frame where the heading turns fastest
and read the speed carried into it and the speed kept through it. Two questions:
  1. Do these in-game break features track separation at the throw (per player, route-adjusted)?
  2. Does the Combine reversal entry speed predict the in-game break entry speed (movement to movement)?
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("comb", HERE / "04_combined.py")
comb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comb)

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/nfl-big-data-bowl-2027")
OUT = HERE.parent / "out"
DT = 0.1
BREAKS = {"OUT", "IN", "CORNER", "POST", "SLANT", "HITCH"}
MIN_ROUTES = 15  # break routes per player in the season


def break_features(xy: np.ndarray, s: np.ndarray) -> dict | None:
    """Fastest heading change after the stem; entry = peak speed in the 1 s before it."""
    if len(s) < 12:
        return None
    d = np.diff(xy, axis=0)
    moving = np.linalg.norm(d, axis=1) > 0.05
    if moving.sum() < 8:
        return None
    head = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # heading change over a 0.3 s window, in deg/s; skip the first 0.8 s (release off the line)
    rate = np.zeros(len(head))
    rate[3:] = np.abs(np.degrees(head[3:] - head[:-3])) / (3 * DT)
    rate[~moving] = 0
    rate[:8] = 0
    i = int(rate.argmax())
    if rate[i] < 90:  # never really broke before the throw
        return None
    lo, hi = max(i - 10, 0), min(i + 4, len(s))
    entry = float(s[lo:i + 1].max())
    if entry < 2.0:
        return None
    through = float(s[max(i - 2, 0):hi].min())
    turn = float(np.degrees(abs(head[min(i + 3, len(head) - 1)] - head[max(i - 6, 0)])))
    return {"entry_speed": entry, "through_speed": through, "retention": through / entry,
            "turn_deg": turn}


def season_breaks(path: Path, keys: set) -> list[dict]:
    """Read one season in chunks, keep only the WR break routes, cut snap -> throw, measure."""
    cols = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, na_values=["NA"], chunksize=4_000_000):
        k = pd.MultiIndex.from_frame(ch[["game_id", "play_id", "nfl_id"]])
        parts.append(ch[k.isin(keys)])
    t = pd.concat(parts)
    ev = t.dropna(subset=["event"]).drop_duplicates(["game_id", "play_id", "event"])
    snap = ev[ev.event == "ball_snap"].set_index(["game_id", "play_id"]).time.rename("t0")
    thr = ev[ev.event == "pass_forward"].set_index(["game_id", "play_id"]).time.rename("t1")
    t = t.join(snap, on=["game_id", "play_id"]).join(thr, on=["game_id", "play_id"]).dropna(
        subset=["t0", "t1"])
    t = t[(t.time >= t.t0) & (t.time <= t.t1)].sort_values("time")
    rows = []
    for (g, pl, pid), d in t.groupby(["game_id", "play_id", "nfl_id"]):
        f = break_features(d[["x", "y"]].to_numpy(), d["s"].to_numpy())
        if f:
            rows.append({"game_id": g, "play_id": pl, "nfl_id": pid, **f})
    print(f"{path.name}: {len(rows)} break routes")
    return rows


def self_check() -> None:
    # 10 yd stem at 7 yd/s, slow to 4 through a 90 deg cut, out at 6 yd/s
    stem = np.c_[np.zeros(15), np.linspace(0, 10, 15)]
    out = np.c_[np.linspace(0.5, 6, 8), np.full(8, 10.0)]
    xy = np.r_[stem, out]
    s = np.r_[np.full(15, 7.0), np.full(8, 4.0)]
    f = break_features(xy, s)
    assert f and abs(f["entry_speed"] - 7) < 1e-9 and f["retention"] < 0.7, f
    assert 60 < f["turn_deg"] < 120, f
    straight = np.c_[np.zeros(25), np.linspace(0, 20, 25)]
    assert break_features(straight, np.full(25, 8.0)) is None


def main() -> None:
    self_check()
    p = pd.read_csv(DATA / "player_play.csv",
                    usecols=["game_id", "play_id", "nfl_id", "route_ran", "separation_at_pass_forward"])
    p = p.dropna(subset=["route_ran", "separation_at_pass_forward"])
    p["sep_over_route"] = p.separation_at_pass_forward - p.groupby("route_ran") \
        .separation_at_pass_forward.transform("mean")
    p = p[p.route_ran.isin(BREAKS)]
    pos = pd.read_csv(DATA / "players.csv")
    wr = set(pos.loc[pos.nfl_position == "WR", "nfl_id"])
    p = p[p.nfl_id.isin(wr)]

    keys = set(p.set_index(["game_id", "play_id", "nfl_id"]).index)
    rows = []
    for path in sorted(DATA.glob("game_tracking_*.csv")):
        rows += season_breaks(path, keys)
    br = pd.DataFrame(rows).merge(p, on=["game_id", "play_id", "nfl_id"])
    br.to_csv(OUT / "game_breaks.csv", index=False)
    print(f"break routes measured={len(br)}  players={br.nfl_id.nunique()}  "
          f"median turn={br.turn_deg.median():.0f} deg")

    # route-adjust the movement features too: a slant and a hitch are run at different speeds
    for f in ["entry_speed", "retention"]:
        br[f + "_adj"] = br[f] - br.groupby("route_ran")[f].transform("mean")
    per = br.groupby("nfl_id").agg(n_routes=("sep_over_route", "size"),
                                   sep_over_route=("sep_over_route", "mean"),
                                   game_entry=("entry_speed_adj", "median"),
                                   game_retention=("retention_adj", "median"))
    per = per[per.n_routes >= MIN_ROUTES]
    combine = pd.read_csv(OUT / "combined_vs_separation.csv", index_col="nfl_id")[
        ["rev_entry_pooled", "composite", "three_cone"]]
    df = per.join(combine, how="left")
    df.to_csv(OUT / "game_breaks_vs_separation.csv")

    rows = []
    pairs = [("game_entry", "sep_over_route"), ("game_retention", "sep_over_route"),
             ("rev_entry_pooled", "game_entry"), ("rev_entry_pooled", "game_retention"),
             ("composite", "game_entry"), ("three_cone", "game_entry")]
    for a, b in pairs:
        d = df.dropna(subset=[a, b])
        if len(d) < 10:
            continue
        rho, lo, hi = comb.spearman_ci(d[a], d[b])
        rows.append({"x": a, "y": b, "n": len(d), "spearman": round(rho, 3),
                     "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "game_breaks_summary.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
