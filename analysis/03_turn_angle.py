"""Split 3-cone / short-shuttle turns by angle: the 180-degree reversals vs the ~90-degree cuts.

02_cod_phases.py found that slicing the median turn does not beat the 3-cone stopwatch. A 3-cone
rep mixes one 180 at cone 2 with ~90/135 bends around cone 3, and the shuttle is all 180s, so the
median blends two different skills. Here each turn gets the angle between the path into the plant
and the path out of it (x, y at 10 Hz), and the phases are compared per class with the same
route-adjusted separation target.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cod", HERE / "02_cod_phases.py")
cod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cod)

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/nfl-big-data-bowl-2027")
OUT = HERE.parent / "out"
FEATURES = ["entry_speed", "decel_rate", "plant_time", "reaccel_time", "turn_time"]
REVERSAL = 135.0  # degrees; above this the turn is a plant-and-go-back, below it a cut


def turn_angle(xy: np.ndarray, pk: int, lo: int, rp: int) -> float:
    """Angle in degrees between displacement peak->plant and plant->back-to-speed."""
    a, b = xy[lo] - xy[pk], xy[rp] - xy[lo]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


BEND_RATE = 60.0  # deg/s of heading change that counts as running a bend


def bends(xy: np.ndarray, s: np.ndarray, rev: list[tuple[int, int]]) -> dict | None:
    """Speed held while bending at speed: frames turning >= BEND_RATE outside the reversals."""
    d = np.diff(xy, axis=0)
    head = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    rate = np.abs(np.degrees(np.gradient(head))) / cod.DT
    mask = (rate >= BEND_RATE) & (s[1:] >= 0.5 * s.max())
    for a, b in rev:
        mask[max(a - 1, 0):b] = False
    if mask.sum() < 3:
        return None
    return {"bend_speed": float(s[1:][mask].mean()), "bend_time": mask.sum() * cod.DT,
            "bend_speed_ratio": float(s[1:][mask].mean() / s.max())}


def self_check() -> None:
    xy = np.array([[0, 0], [5, 0], [5, 5], [0, 0.0]])
    assert abs(turn_angle(xy, 0, 1, 2) - 90) < 1e-6
    assert abs(turn_angle(np.array([[0, 0], [5, 0], [0, 0.0]]), 0, 1, 2) - 180) < 1e-6
    th = np.radians(np.arange(0, 180, 9.0))  # half circle at 90 deg/s, constant speed
    b = bends(np.c_[np.cos(th), np.sin(th)], np.full(len(th), 5.0), [])
    assert b and abs(b["bend_speed_ratio"] - 1) < 1e-9, b
    assert bends(np.c_[np.arange(20.0), np.zeros(20)], np.full(20, 5.0), []) is None


def main() -> None:
    self_check()
    c = pd.read_csv(DATA / "combine_tracking.csv")
    c = c[(c.entity_type == "PLAYER") & c.drill_name.isin(cod.DRILLS)].sort_values("time")
    rows, brows = [], []
    for (pid, drill, att), g in c.groupby(["nfl_id", "drill_name", "attempt"]):
        xy = g[["x", "y"]].to_numpy()
        ts = cod.turns(g["s"].to_numpy())
        if drill == "THREE_CONE_DRILL":
            b = bends(xy, g["s"].to_numpy(), [(t["i_peak"], t["i_back"]) for t in ts])
            if b:
                brows.append({"nfl_id": pid, **b})
        for t in ts:
            ang = turn_angle(xy, t["i_peak"], t["i_low"], t["i_back"])
            rows.append({"nfl_id": pid, "drill": drill, "angle": ang,
                         **{f: t[f] for f in FEATURES}})
    tr = pd.DataFrame(rows).dropna(subset=["angle"])
    tr["kind"] = np.where(tr.angle >= REVERSAL, "reversal", "cut")
    print(tr.groupby(["drill", "kind"]).angle.agg(["count", "median"]).round(1).to_string())

    p = pd.read_csv(DATA / "player_play.csv",
                    usecols=["nfl_id", "route_ran", "separation_at_pass_forward"]).dropna()
    p["sep_over_route"] = p.separation_at_pass_forward - p.groupby("route_ran") \
        .separation_at_pass_forward.transform("mean")
    game = p.groupby("nfl_id").sep_over_route.agg(["mean", "count"]).rename(
        columns={"mean": "sep_over_route", "count": "n_targets"})
    game = game[game.n_targets >= 30]

    out = []
    for kind, g in tr.groupby("kind"):
        per = g.groupby("nfl_id")[FEATURES].median().join(g.groupby("nfl_id").size().rename("n"))
        df = per.join(game, how="inner")
        for f in FEATURES:
            r = df[[f, "sep_over_route"]].corr(method="spearman").iloc[0, 1]
            out.append({"kind": kind, "feature": f, "n_players": len(df), "spearman": round(r, 3)})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "turn_angle_vs_separation.csv", index=False)
    print(res.pivot(index="feature", columns="kind", values="spearman").to_string())
    print("n_players:", res.groupby("kind").n_players.first().to_dict())

    bd = pd.DataFrame(brows).groupby("nfl_id").median().join(game, how="inner")
    res3 = pd.read_csv(DATA / "combine_results.csv")[["nfl_id", "three_cone"]].set_index("nfl_id")
    bd = bd.join(res3)
    bd.to_csv(OUT / "three_cone_bends_vs_separation.csv")
    for f in ["bend_speed", "bend_speed_ratio", "bend_time", "three_cone"]:
        d = bd.dropna(subset=[f])
        r = d[[f, "sep_over_route"]].corr(method="spearman").iloc[0, 1]
        print(f"3-cone {f:>16}  n={len(d):>3}  spearman_vs_separation={r:+.3f}")
    d = bd.dropna(subset=["three_cone"])
    print("bend_speed vs stopwatch 3-cone spearman:",
          round(d[["bend_speed", "three_cone"]].corr(method="spearman").iloc[0, 1], 3), "n=", len(d))


if __name__ == "__main__":
    main()
