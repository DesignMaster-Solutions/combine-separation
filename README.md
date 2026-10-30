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

## Run
```
kaggle competitions download nfl-big-data-bowl-2027 -p data
python analysis/01_break_retention.py data/nfl-big-data-bowl-2027
python analysis/02_cod_phases.py data/nfl-big-data-bowl-2027
```
Requires pandas and numpy.

Design Master Solutions LLC · MIT
