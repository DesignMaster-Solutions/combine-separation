"""Combine 10 Hz route drills -> NFL separation.

For every WR route drill with a hard cut, measure how much speed the player keeps
through the break (speed at the sharpest heading change / peak speed before it).
Average per player, then compare with route-adjusted separation at pass forward
in 2023-25 games (player_play.csv, derived from the 10 Hz game tracking).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/nfl-big-data-bowl-2027")
OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)

BREAK_DRILLS = ["SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT", "CURL_ROUTE_RIGHT",
                "COMEBACK_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT"]


def break_retention(g: pd.DataFrame) -> float:
    g = g.sort_values("time")
    s = g["s"].to_numpy()
    d = np.unwrap(np.radians(g["dir"].to_numpy()))
    if len(s) < 15 or s.max() < 3:
        return np.nan
    turn = np.abs(np.gradient(pd.Series(d).rolling(3, center=True, min_periods=1).mean()))
    turn[s < 1.5] = 0  # heading is noise when nearly stopped
    i = int(turn.argmax())
    pre = s[: i + 1].max()
    return float(s[i] / pre) if pre > 0 else np.nan


def main() -> None:
    c = pd.read_csv(DATA / "combine_tracking.csv")
    c = c[(c.entity_type == "PLAYER") & c.drill_name.isin(BREAK_DRILLS)]
    att = (c.groupby(["nfl_id", "drill_name", "attempt"]).apply(break_retention, include_groups=False)
           .rename("retention").dropna().reset_index())
    per_player = att.groupby("nfl_id").retention.agg(["mean", "count"]).rename(
        columns={"mean": "break_retention", "count": "n_reps"})

    p = pd.read_csv(DATA / "player_play.csv",
                    usecols=["nfl_id", "route_ran", "separation_at_pass_forward"])
    p = p.dropna()
    p["sep_over_route"] = p.separation_at_pass_forward - p.groupby("route_ran") \
        .separation_at_pass_forward.transform("mean")
    game = p.groupby("nfl_id").sep_over_route.agg(["mean", "count"]).rename(
        columns={"mean": "sep_over_route", "count": "n_targets"})

    res = pd.read_csv(DATA / "combine_results.csv")[["nfl_id", "three_cone", "forty"]]
    df = per_player.join(game, how="inner").reset_index().merge(res, on="nfl_id", how="left")
    df = df[(df.n_reps >= 3) & (df.n_targets >= 30)]
    df.to_csv(OUT / "break_retention_vs_separation.csv", index=False)

    r = df[["break_retention", "sep_over_route"]].corr(method="spearman").iloc[0, 1]
    tc = df.dropna(subset=["three_cone"])
    r3 = tc[["three_cone", "sep_over_route"]].corr(method="spearman").iloc[0, 1]
    print(f"players={len(df)}  spearman(break_retention, separation)={r:.3f}")
    print(f"players_with_3cone={len(tc)}  spearman(three_cone, separation)={r3:.3f}")


if __name__ == "__main__":
    main()
