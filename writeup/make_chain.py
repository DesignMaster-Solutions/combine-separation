"""Draw writeup/chain.png from the summary CSVs written by analysis/04-08."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "out"


def pick(csv, **where):
    df = pd.read_csv(OUT / csv)
    for k, v in where.items():
        df = df[df[k] == v]
    assert len(df) == 1, (csv, where, len(df))
    r = df.iloc[0]
    return r.spearman, r.ci90_lo, r.ci90_hi


ROWS = [
    ("Combine cut entry -> game break entry", pick("combine_vs_game_summary.csv", x="combine_entry", y="game_entry")),
    ("Combine kept speed -> game kept speed", pick("combine_vs_game_summary.csv", x="combine_retention", y="game_retention")),
    ("Game break entry -> separation", pick("game_breaks_summary.csv", x="game_entry", y="sep_over_route")),
    ("Game kept speed -> separation", pick("game_breaks_summary.csv", x="game_retention", y="sep_over_route")),
    ("Kept speed -> separation, man minus zone", pick("coverage_split_summary.csv", level="route", coverage="man_minus_zone")),
    ("Kept speed -> separation, defender within 5 yd", pick("defender_at_break_summary.csv", subset="matchup", x="ret_adj", y="sep_over_route")),
]


def main(path):
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    for i, (label, (rho, lo, hi)) in enumerate(reversed(ROWS)):
        color = "#1f6f50" if lo > -0.005 else "#9a9a9a"
        ax.errorbar(rho, i, xerr=[[rho - lo], [hi - rho]], fmt="o", color=color, capsize=3)
        ax.text(-0.62, i, label, va="center", ha="left", fontsize=8)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlim(-0.65, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("Spearman with 90% bootstrap interval (green: interval at or above zero)")
    ax.set_title("Arrival speed repeats but does not get receivers open; kept speed does, and the Combine misses it", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "chain.png")
