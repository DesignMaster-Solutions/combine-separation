"""In man coverage, is the kept speed at the break the cause of separation, or do defenders just give room?

07_coverage_split.py found that the speed a WR keeps through the break goes with separation in man
(+0.08 per route) and barely in zone (+0.02). Two stories fit that: (a) keeping speed through the cut
beats the trailing defender, or (b) the receivers who keep speed are the ones a defender already
gave room to, so they never had to slow down. This re-reads the tracking of the man plays only, finds
the same break frame as 05_game_breaks.py, and measures the nearest defender there:
  - dist_at_break: yards from the WR to the nearest defender at the break frame
  - leverage: cosine between the WR's post-break heading and the direction to that defender
    (> 0 the defender sits where the WR is going, < 0 he trails behind the cut)
  - sep_gain: separation at the throw minus dist_at_break (space made after the cut)
Story (b) predicts kept speed goes with dist_at_break and not with sep_gain; story (a) the reverse.
Caveat found on the first run: the tracking files carry only Combine participants (1-3 players per
play), not all 22. The "nearest defender" is the nearest TRACKED one, often a safety 15 yd away, so the
defender tests use only the matchup routes where that defender is within MATCHUP_YD at the break
(most likely the man defender on that receiver).
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
cs = load("cs", "07_coverage_split.py")

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../data/nfl-big-data-bowl-2027")
OUT = HERE.parent / "out"
RNG = np.random.default_rng(11)
MATCHUP_YD = 5.0


def defender_at_break(wr_xy: np.ndarray, i: int, defs: np.ndarray) -> tuple[float, float]:
    """Nearest defender distance at frame i and the cosine of his direction vs the post-break heading.

    defs: (n_defenders, 2) positions at frame i."""
    rel = defs - wr_xy[i]
    dist = np.linalg.norm(rel, axis=1)
    k = int(dist.argmin())
    head = wr_xy[min(i + 3, len(wr_xy) - 1)] - wr_xy[i]
    norm = np.linalg.norm(head) * dist[k]
    lev = float(rel[k] @ head / norm) if norm > 0 else 0.0
    return float(dist[k]), lev


def route_boot(df: pd.DataFrame, x: str, y: str, draws: int = 1000) -> tuple[float, float, float]:
    """Route-level Spearman with a 90% interval resampling players (routes cluster by receiver)."""
    groups = [g for _, g in df.groupby("nfl_id")]
    rho = cs.rank_corr(df[x], df[y])
    boot = []
    for _ in range(draws):
        d = pd.concat([groups[k] for k in RNG.integers(0, len(groups), len(groups))])
        boot.append(cs.rank_corr(d[x], d[y]))
    lo, hi = np.percentile(boot, [5, 95])
    return rho, float(lo), float(hi)


def self_check() -> None:
    # WR runs up the y axis then cuts to +x at frame 5
    xy = np.r_[np.c_[np.zeros(6), np.arange(6.0)], np.c_[np.arange(1, 5.0), np.full(4, 5.0)]]
    d, lev = defender_at_break(xy, 5, np.array([[3.0, 5.0], [0.0, 20.0]]))
    assert abs(d - 3) < 1e-9 and lev > 0.99, (d, lev)  # defender sitting on the break side
    d, lev = defender_at_break(xy, 5, np.array([[-2.0, 5.0]]))
    assert abs(d - 2) < 1e-9 and lev < -0.99, (d, lev)  # defender trailing behind the cut


def man_tracking(path: Path, plays: set) -> pd.DataFrame:
    cols = ["game_id", "play_id", "nfl_id", "time", "x", "y", "s", "event"]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, na_values=["NA"], chunksize=4_000_000):
        k = pd.MultiIndex.from_frame(ch[["game_id", "play_id"]])
        parts.append(ch[k.isin(plays)])
    return pd.concat(parts)


def measure(t: pd.DataFrame, routes: pd.DataFrame, side: pd.DataFrame) -> list[dict]:
    ev = t.dropna(subset=["event"]).drop_duplicates(["game_id", "play_id", "event"])
    snap = ev[ev.event == "ball_snap"].set_index(["game_id", "play_id"]).time.rename("t0")
    thr = ev[ev.event == "pass_forward"].set_index(["game_id", "play_id"]).time.rename("t1")
    t = t.join(snap, on=["game_id", "play_id"]).join(thr, on=["game_id", "play_id"]).dropna(
        subset=["t0", "t1"])
    t = t[(t.time >= t.t0) & (t.time <= t.t1)].merge(side, on=["game_id", "play_id", "nfl_id"])
    rows = []
    want = routes.set_index(["game_id", "play_id"]).nfl_id
    for (g, pl), d in t.groupby(["game_id", "play_id"]):
        if (g, pl) not in want.index:
            continue
        for wr in np.atleast_1d(want.loc[(g, pl)]):
            w = d[d.nfl_id == wr].sort_values("time")
            f = gb.break_features(w[["x", "y"]].to_numpy(), w["s"].to_numpy())
            if not f:
                continue
            tb = w.time.iloc[f["break_frame"]]
            defs = d[(d.defense == 1) & (d.time == tb)][["x", "y"]].to_numpy()
            if len(defs) == 0:
                continue
            dist, lev = defender_at_break(w[["x", "y"]].to_numpy(), f["break_frame"], defs)
            rows.append({"game_id": g, "play_id": pl, "nfl_id": wr, "dist_at_break": dist, "leverage": lev})
    print(f"{len(rows)} man routes with a defender at the break")
    return rows


def main() -> None:
    self_check()
    br = pd.read_csv(OUT / "game_breaks.csv")
    pp = pd.read_csv(DATA / "player_play.csv", na_values=["NA"],
                     usecols=["game_id", "play_id", "nfl_id", "team_abbr", "possession_team",
                              "team_coverage_man_zone", "cushion"])
    cov = pp[pp.team_abbr != pp.possession_team].dropna(subset=["team_coverage_man_zone"]) \
        .drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", "team_coverage_man_zone"]]
    man = br.merge(cov[cov.team_coverage_man_zone == "MAN_COVERAGE"], on=["game_id", "play_id"])
    man = man.merge(pp[["game_id", "play_id", "nfl_id", "cushion"]], on=["game_id", "play_id", "nfl_id"])
    man = man.dropna(subset=["cushion"])
    side = pp[["game_id", "play_id", "nfl_id"]].assign(defense=(pp.team_abbr != pp.possession_team).astype(int))
    plays = set(man.set_index(["game_id", "play_id"]).index)

    cache = OUT / "defender_at_break_raw.csv"  # shortcut: delete it when game_breaks.csv changes
    if cache.exists():
        raw = pd.read_csv(cache)
    else:
        rows = []
        for path in sorted(DATA.glob("game_tracking_*.csv")):
            rows += measure(man_tracking(path, plays), man, side)
        raw = pd.DataFrame(rows)
        raw.to_csv(cache, index=False)
    df = man.merge(raw, on=["game_id", "play_id", "nfl_id"])
    df["ret_adj"] = df.retention - df.groupby("route_ran").retention.transform("mean")
    df["sep_gain"] = df.separation_at_pass_forward - df.dist_at_break
    df["sep_gain_adj"] = df.sep_gain - df.groupby("route_ran").sep_gain.transform("mean")
    df["dist_adj"] = df.dist_at_break - df.groupby("route_ran").dist_at_break.transform("mean")
    df["sep_given_dist"] = cs.resid(df.sep_over_route, df.dist_adj)
    df.to_csv(OUT / "defender_at_break.csv", index=False)
    print(f"man break routes={len(df)}  players={df.nfl_id.nunique()}  "
          f"median dist at break={df.dist_at_break.median():.2f} yd  "
          f"defender on the break side={(df.leverage > 0).mean():.0%}")
    mu = df[df.dist_at_break < MATCHUP_YD]
    print(f"matchup routes (tracked defender < {MATCHUP_YD:.0f} yd at the break)={len(mu)}  "
          f"players={mu.nfl_id.nunique()}  defender on the break side={(mu.leverage > 0).mean():.0%}")

    res = []
    tests = [("all man", df, "ret_adj", "sep_over_route"),
             ("matchup", mu, "ret_adj", "dist_adj"), ("matchup", mu, "ret_adj", "sep_gain_adj"),
             ("matchup", mu, "ret_adj", "sep_given_dist"), ("matchup", mu, "ret_adj", "sep_over_route"),
             ("matchup, defender on break side", mu[mu.leverage > 0], "ret_adj", "sep_over_route"),
             ("matchup, defender trailing", mu[mu.leverage <= 0], "ret_adj", "sep_over_route")]
    for name, d, x, y in tests:
        rho, lo, hi = route_boot(d, x, y)
        res.append({"subset": name, "x": x, "y": y, "n": len(d), "spearman": round(rho, 3),
                    "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)})
    res = pd.DataFrame(res)
    res.to_csv(OUT / "defender_at_break_summary.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
