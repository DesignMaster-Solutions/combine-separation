# Does the Combine see separation? (NFL Big Data Bowl 2027)

Working repo for our Big Data Bowl 2027 Open Track entry (Kaggle writeup + public notebook,
deadline 2027-01-06). Code only: competition data is CC BY-NC and is **not** stored here.

## Question
Wide receivers are drafted partly on Combine drills. The 2027 data adds 10 Hz tracking of those
drills (40, 3-cone, shuttle, position route drills). Which Combine movement traits, measured
frame by frame, carry over to how much separation the same player creates on NFL routes
(10 Hz game tracking 2023-25)?

## Status (2026-10-29)
`analysis/01_break_retention.py`, 93 WRs with >= 3 Combine break-route reps and >= 30 routes
with separation recorded, separation adjusted for route type (Spearman):

| Combine feature | n | rho vs separation |
|---|---|---|
| speed kept through the break, route drills (10 Hz) | 93 | -0.07 |
| peak speed in route drills (10 Hz) | 93 | -0.09 |
| 40-yard time (stopwatch) | 84 | -0.22 |
| 3-cone time (stopwatch) | 28 | -0.43 |

First read: a naive "speed through the cut" from the route drills does not carry over, while the
timed change-of-direction drill does (faster 3-cone, more separation). Next: segment the 3-cone and
shuttle at 10 Hz (decel distance, plant time, re-acceleration) and measure the same phases on NFL
breaks from `game_tracking_*.csv`, so the link is movement-to-movement instead of stopwatch-to-outcome.

### 10 Hz change-of-direction phases (`analysis/02_cod_phases.py`)
3-cone and short-shuttle reps split into turns (546 turns), each turn into deceleration, plant
(time below half the entry speed) and re-acceleration (back to 90% of entry speed). Median turn per
player, 42 players with tracking and >= 30 targets:

| phase feature | n | rho vs separation |
|---|---|---|
| entry speed into the turn | 42 | +0.21 |
| re-acceleration time | 42 | -0.15 |
| deceleration rate | 42 | -0.13 |
| plant time | 42 | -0.03 |
| whole turn time | 42 | -0.02 |
| 3-cone time (stopwatch, same sample) | 32 | -0.36 |
| short shuttle time (stopwatch) | 39 | -0.02 |

Second read: splitting the turn into phases does not beat the stopwatch; the 3-cone signal holds on
a slightly larger sample and the shuttle carries none. The 3-cone has three turns at different
angles while our segmentation pools them, so the next step is per-turn-angle features (the 180 deg
turn vs the two 90 deg ones) before moving to NFL breaks.

## Third pass: turn angle (analysis/03_turn_angle.py)
Each detected turn gets the angle between the path into the plant and the path out of it. Every
detected turn is a reversal (shuttle median 173 deg, 3-cone 154 deg); only 7 of 211 3-cone turns
fall under 135 deg. The ~90 deg bends of the 3-cone are run without the speed ever dropping below
60% of entry, so the speed-minimum detector never sees them. They are measured separately: frames
outside the reversals turning at >= 60 deg/s while above half of the rep's peak speed.

| 3-cone feature (WRs, >= 30 targets) | n | Spearman vs separation |
|---|---|---|
| reversal entry speed (3-cone + shuttle pooled, see fourth pass) | 53 | +0.30 |
| reversal plant time | 53 | +0.14 |
| bend speed / rep peak speed | 37 | +0.19 |
| bend speed (yd/s) | 37 | +0.12 |
| time spent bending | 37 | -0.03 |
| 3-cone stopwatch, same sample | 34 | -0.42 |

Third read: speed carried into the reversal (+0.30, n=53) is the strongest tracking feature so far
and covers more players than the stopwatch. Holding speed through the bends points the right way
(+0.19) but is weak, and it only correlates -0.25 with the stopwatch, so the clock is capturing
something the bends alone do not. Next: combine reversal entry speed with the bend ratio and test
it on the players with no recorded 3-cone time, then measure the same two features on route breaks
in game tracking.

## Fourth pass: composite and the players with no stopwatch (analysis/04_combined.py)
Correction to the third pass: its reversal row pooled 3-cone and shuttle reversals. Split by drill,
3-cone reversals alone give +0.22 (n=37) and shuttle reversals +0.09 (n=42); pooling is what
reaches +0.30, because it covers more players. Here the pooled reversal entry speed and the bend
ratio are z-scored and averaged, and every Spearman carries a 2,000-draw bootstrap 90% interval.

| feature (WRs, >= 30 targets) | subset | n | Spearman | 90% CI |
|---|---|---|---|---|
| composite (reversal entry + bend ratio) | all | 53 | +0.32 | +0.09 to +0.52 |
| reversal entry speed, pooled | all | 53 | +0.30 | +0.07 to +0.50 |
| 3-cone stopwatch | all | 39 | -0.43 | -0.60 to -0.20 |
| reversal entry speed, pooled | no timed 3-cone | 14 | +0.50 | +0.05 to +0.78 |
| composite | no timed 3-cone | 14 | +0.36 | -0.17 to +0.75 |
| composite | with timed 3-cone | 39 | +0.29 | +0.01 to +0.53 |

Fourth read: the composite is the first tracking measure whose interval clears zero on the full
sample, and the reversal entry speed keeps its sign on the 14 receivers the stopwatch misses
(small n, wide interval). The stopwatch is still the single best number where it exists, so the
tracking features are a fill-in for missing times, not a replacement. Next: measure entry speed
into route breaks in game tracking and check it against the same separation target.

## Run
```
kaggle competitions download nfl-big-data-bowl-2027 -p data
python analysis/01_break_retention.py data/nfl-big-data-bowl-2027
python analysis/02_cod_phases.py data/nfl-big-data-bowl-2027
python analysis/03_turn_angle.py data/nfl-big-data-bowl-2027
python analysis/04_combined.py data/nfl-big-data-bowl-2027
```
Requires pandas and numpy.

Design Master Solutions LLC · MIT
