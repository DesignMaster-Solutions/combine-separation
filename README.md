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

## Run
```
kaggle competitions download nfl-big-data-bowl-2027 -p data
python analysis/01_break_retention.py data/nfl-big-data-bowl-2027
```
Requires pandas and numpy.

Design Master Solutions LLC · MIT
