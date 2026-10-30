"""Combine the two best 3-cone tracking features and test them where the stopwatch is missing.

03_turn_angle.py found speed carried into the reversal (+0.30) and speed held through the bends
(+0.19) as the strongest tracking signals. Here both are z-scored and averaged per player, with
the reversal measured per drill (03 pooled 3-cone and shuttle reversals), and every Spearman gets
a 2,000-draw bootstrap 90% interval. The question that matters for scouting: does the composite
still carry signal on WRs who never ran a timed 3-cone?
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cod, ang = load("cod", "02_cod_phases.py"), load("ang", "03_turn_angle.py")
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/nfl-big-data-bowl-2027")
OUT = HERE.parent / "out"
RNG = np.random.default_rng(27)


def spearman_ci(x: pd.Series, y: pd.Series, draws: int = 2000) -> tuple[float, float, float]:
    """Spearman rho with a percentile bootstrap 90% interval over players."""
    rx, ry = x.rank().to_numpy(), y.rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n, boot = len(x), []
    for _ in range(draws):
        i = RNG.integers(0, n, n)
        bx, by = pd.Series(x.to_numpy()[i]).rank(), pd.Series(y.to_numpy()[i]).rank()
        if bx.std() > 0 and by.std() > 0:
            boot.append(np.corrcoef(bx, by)[0, 1])
    lo, hi = np.percentile(boot, [5, 95])
    return rho, float(lo), float(hi)


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def self_check() -> None:
    x = pd.Series(np.arange(40.0))
    rho, lo, hi = spearman_ci(x, x * 2 + 1, draws=200)
    assert abs(rho - 1) < 1e-9 and lo > 0.99, (rho, lo)
    rho, lo, hi = spearman_ci(x, -x, draws=200)
    assert abs(rho + 1) < 1e-9 and hi < -0.99
    assert abs(z(pd.Series([1.0, 2, 3])).mean()) < 1e-12


def main() -> None:
    self_check()
    c = pd.read_csv(DATA / "combine_tracking.csv")
    c = c[(c.entity_type == "PLAYER") & c.drill_name.isin(cod.DRILLS)].sort_values("time")
    rev, bnd = [], []
    for (pid, drill, att), g in c.groupby(["nfl_id", "drill_name", "attempt"]):
        xy, s = g[["x", "y"]].to_numpy(), g["s"].to_numpy()
        ts = cod.turns(s)
        for t in ts:
            if ang.turn_angle(xy, t["i_peak"], t["i_low"], t["i_back"]) >= ang.REVERSAL:
                rev.append({"nfl_id": pid, "drill": drill, "entry_speed": t["entry_speed"]})
        if drill == "THREE_CONE_DRILL":
            b = ang.bends(xy, s, [(t["i_peak"], t["i_back"]) for t in ts])
            if b:
                bnd.append({"nfl_id": pid, "bend_speed_ratio": b["bend_speed_ratio"]})
    rv = pd.DataFrame(rev).pivot_table(index="nfl_id", columns="drill", values="entry_speed",
                                       aggfunc="median")
    rv.columns = ["rev_entry_" + ("3cone" if k.startswith("THREE") else "shuttle") for k in rv.columns]
    rv["rev_entry_pooled"] = pd.DataFrame(rev).groupby("nfl_id").entry_speed.median()
    per = rv.join(pd.DataFrame(bnd).groupby("nfl_id").median(), how="outer")

    p = pd.read_csv(DATA / "player_play.csv",
                    usecols=["nfl_id", "route_ran", "separation_at_pass_forward"]).dropna()
    p["sep_over_route"] = p.separation_at_pass_forward - p.groupby("route_ran") \
        .separation_at_pass_forward.transform("mean")
    game = p.groupby("nfl_id").sep_over_route.agg(["mean", "count"]).rename(
        columns={"mean": "sep_over_route", "count": "n_targets"})
    df = per.join(game[game.n_targets >= 30], how="inner").join(
        pd.read_csv(DATA / "combine_results.csv")[["nfl_id", "three_cone"]].set_index("nfl_id"))
    df["composite"] = pd.concat([z(df.rev_entry_pooled), z(df.bend_speed_ratio)], axis=1).mean(axis=1)
    df.to_csv(OUT / "combined_vs_separation.csv")

    feats = ["rev_entry_3cone", "rev_entry_shuttle", "rev_entry_pooled", "bend_speed_ratio",
             "composite", "three_cone"]
    rows = []
    for subset, d0 in [("all", df), ("no_stopwatch_3cone", df[df.three_cone.isna()]),
                       ("with_stopwatch_3cone", df[df.three_cone.notna()])]:
        for f in feats:
            d = d0.dropna(subset=[f])
            if len(d) < 10:
                continue
            rho, lo, hi = spearman_ci(d[f], d.sep_over_route)
            rows.append({"subset": subset, "feature": f, "n": len(d), "spearman": round(rho, 3),
                         "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "combined_summary.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
