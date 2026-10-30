"""Same break detector on both sides: Combine route drills vs NFL game routes.

05_game_breaks.py measured the route break in 2023-25 game tracking and found that the speed a WR
keeps through the cut tracks route-adjusted separation (+0.21). It also found that the Combine
*reversal* entry speed does not predict the in-game break entry. Reversals (180 deg) are not route
cuts, so here the exact `break_features` of 05 runs on the Combine route drills (slant, dagger,
curl, comeback, slot strike), drill-adjusted, and is compared movement-to-movement with the game.
Question: does the speed a WR keeps through a Combine route cut predict the speed he keeps through
real route cuts, and does it reach separation directly?
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gb = load("gb", "05_game_breaks.py")
comb = gb.comb

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/nfl-big-data-bowl-2027")
OUT = HERE.parent / "out"
ROUTE_DRILLS = ["SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT", "CURL_ROUTE_RIGHT",
                "COMEBACK_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT"]
MIN_REPS = 3


def combine_breaks(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, drill, att), d in c.groupby(["nfl_id", "drill_name", "attempt"]):
        d = d.sort_values("time")
        f = gb.break_features(d[["x", "y"]].to_numpy(), d["s"].to_numpy())
        if f:
            rows.append({"nfl_id": pid, "drill_name": drill, "attempt": att, **f})
    br = pd.DataFrame(rows)
    for f in ["entry_speed", "retention"]:
        br[f + "_adj"] = br[f] - br.groupby("drill_name")[f].transform("mean")
    return br


def self_check() -> None:
    # two players, one drill: the one who keeps more speed through the cut gets the higher adj retention
    stem = np.c_[np.zeros(15), np.linspace(0, 10, 15)]
    out = np.c_[np.linspace(0.5, 6, 8), np.full(8, 10.0)]
    xy = np.r_[stem, out]
    rows = []
    for pid, kept in [(1, 6.0), (2, 3.0)]:
        s = np.r_[np.full(15, 7.0), np.full(8, kept)]
        rows.append(pd.DataFrame({"nfl_id": pid, "drill_name": "SLANT_ROUTE_LEFT", "attempt": 1,
                                  "time": np.arange(len(s)), "x": xy[:, 0], "y": xy[:, 1], "s": s}))
    br = combine_breaks(pd.concat(rows)).set_index("nfl_id")
    assert br.loc[1, "retention_adj"] > 0 > br.loc[2, "retention_adj"], br


def main() -> None:
    self_check()
    c = pd.read_csv(DATA / "combine_tracking.csv",
                    usecols=["nfl_id", "entity_type", "time", "drill_name", "attempt", "x", "y", "s"])
    c = c[(c.entity_type == "PLAYER") & c.drill_name.isin(ROUTE_DRILLS)]
    br = combine_breaks(c)
    print(f"combine route cuts measured={len(br)}  players={br.nfl_id.nunique()}  "
          f"median turn={br.turn_deg.median():.0f} deg")
    per = br.groupby("nfl_id").agg(n_reps=("retention", "size"),
                                   combine_entry=("entry_speed_adj", "median"),
                                   combine_retention=("retention_adj", "median"))
    per = per[per.n_reps >= MIN_REPS]

    game = pd.read_csv(OUT / "game_breaks_vs_separation.csv", index_col="nfl_id")[
        ["n_routes", "sep_over_route", "game_entry", "game_retention"]]
    df = per.join(game, how="inner")
    df.to_csv(OUT / "combine_vs_game_retention.csv")

    rows = []
    pairs = [("combine_retention", "game_retention"), ("combine_entry", "game_entry"),
             ("combine_retention", "sep_over_route"), ("combine_entry", "sep_over_route"),
             ("game_retention", "sep_over_route")]
    for a, b in pairs:
        d = df.dropna(subset=[a, b])
        if len(d) < 10:
            continue
        rho, lo, hi = comb.spearman_ci(d[a], d[b])
        rows.append({"x": a, "y": b, "n": len(d), "spearman": round(rho, 3),
                     "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "combine_vs_game_summary.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
