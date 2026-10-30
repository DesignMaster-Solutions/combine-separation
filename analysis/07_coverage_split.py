"""Does the kept speed at the break pay more against man coverage than against zone?

05_game_breaks.py found that the speed a WR keeps through the break tracks route-adjusted separation
(+0.21 per player). Against man the defender trails the receiver, so the cut itself should create the
separation; against zone the defender sits in an area and separation comes from where the route ends.
If the kept speed is real route craft, its link to separation should be stronger in man. This reuses
out/game_breaks.csv (no tracking re-read) and adds the defense's man/zone call and the cushion at the
snap from player_play.csv. Two readings:
  1. Route level: retention (route-adjusted) vs separation over route, within man and within zone,
     controlling for cushion (residualised), with a bootstrap over players for the gap man - zone.
  2. Player level: one retention per WR (all breaks) vs his mean separation in man routes and in zone
     routes separately.
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
RNG = np.random.default_rng(7)
MIN_ROUTES = 8  # break routes per player inside one coverage family


def resid(y: pd.Series, x: pd.Series) -> pd.Series:
    """y minus its least-squares line on x (removes the cushion effect)."""
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x)


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a.rank(), b.rank())[0, 1])


def gap_by_player(df: pd.DataFrame, draws: int = 1000) -> tuple[float, float, float]:
    """rho(man) - rho(zone) at route level, 90% interval resampling players (routes are clustered)."""
    def gap(d: pd.DataFrame) -> float:
        m, z = d[d["cov"] == "MAN_COVERAGE"], d[d["cov"] == "ZONE_COVERAGE"]
        return rank_corr(m.ret_r, m.sep_r) - rank_corr(z.ret_r, z.sep_r)
    groups = {k: g for k, g in df.groupby("nfl_id")}
    ids = np.array(list(groups))
    boot = [gap(pd.concat([groups[k] for k in RNG.choice(ids, len(ids))])) for _ in range(draws)]
    lo, hi = np.percentile(boot, [5, 95])
    return gap(df), float(lo), float(hi)


def self_check() -> None:
    # retention drives separation only in man: the gap must come out clearly positive
    n = 4000
    r = np.random.default_rng(1)
    d = pd.DataFrame({"nfl_id": r.integers(0, 80, n), "cov": r.choice(["MAN_COVERAGE", "ZONE_COVERAGE"], n),
                      "ret_r": r.normal(size=n)})
    d["sep_r"] = np.where(d["cov"] == "MAN_COVERAGE", d.ret_r, 0) + r.normal(size=n)
    g, lo, _ = gap_by_player(d, draws=200)
    assert g > 0.4 and lo > 0.3, (g, lo)
    x = pd.Series(np.arange(10.0))
    assert np.allclose(resid(2 * x + 1, x), 0)


def main() -> None:
    self_check()
    br = pd.read_csv(OUT / "game_breaks.csv")
    pp = pd.read_csv(DATA / "player_play.csv", na_values=["NA"],
                     usecols=["game_id", "play_id", "nfl_id", "team_abbr", "possession_team",
                              "team_coverage_man_zone", "cushion"])
    cov = pp[pp.team_abbr != pp.possession_team].dropna(subset=["team_coverage_man_zone"]) \
        .drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", "team_coverage_man_zone"]]
    wr = pp[["game_id", "play_id", "nfl_id", "cushion"]]
    df = br.merge(cov, on=["game_id", "play_id"]).merge(wr, on=["game_id", "play_id", "nfl_id"])
    df = df.rename(columns={"team_coverage_man_zone": "cov"}).dropna(subset=["cushion"])
    df = df[df["cov"].isin(["MAN_COVERAGE", "ZONE_COVERAGE"])].copy()
    df["ret_adj"] = df.retention - df.groupby("route_ran").retention.transform("mean")
    df["ret_r"] = resid(df.ret_adj, df.cushion)
    df["sep_r"] = resid(df.sep_over_route, df.cushion)
    print(f"break routes with coverage call={len(df)}  man={int((df['cov'] == 'MAN_COVERAGE').sum())}  "
          f"zone={int((df['cov'] == 'ZONE_COVERAGE').sum())}  players={df.nfl_id.nunique()}")

    rows = []
    for c, d in df.groupby("cov"):
        rows.append({"level": "route", "coverage": c, "n": len(d),
                     "spearman": round(rank_corr(d.ret_r, d.sep_r), 3)})
    g, lo, hi = gap_by_player(df)
    rows.append({"level": "route", "coverage": "man_minus_zone", "n": len(df), "spearman": round(g, 3),
                 "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)})

    ret = df.groupby("nfl_id").ret_adj.median().rename("retention")
    for c, d in df.groupby("cov"):
        per = d.groupby("nfl_id").agg(n=("sep_r", "size"), sep=("sep_r", "mean"))
        per = per[per.n >= MIN_ROUTES].join(ret)
        rho, lo, hi = comb.spearman_ci(per.retention, per.sep)
        rows.append({"level": "player", "coverage": c, "n": len(per), "spearman": round(rho, 3),
                     "ci90_lo": round(lo, 3), "ci90_hi": round(hi, 3)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "coverage_split_summary.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
