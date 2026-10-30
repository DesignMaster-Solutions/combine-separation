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
python analysis/05_game_breaks.py data/nfl-big-data-bowl-2027
python analysis/06_combine_game_retention.py data/nfl-big-data-bowl-2027
python analysis/07_coverage_split.py data/nfl-big-data-bowl-2027
```
Requires pandas and numpy.

Design Master Solutions LLC · MIT

## Fifth pass: the break on the field (analysis/05_game_breaks.py)
Game tracking 2023-25 at 10 Hz, WR routes with a break (out, in, corner, post, slant, hitch), snap to
throw. The break is the frame where the heading turns fastest (0.3 s window, after the first 0.8 s);
entry = peak speed in the second before it, retention = slowest speed through it / entry. Both are
route-adjusted (a hitch and a slant are run at different speeds) and taken as a per-player median.
3,514 break routes, 99 WRs, median turn 81 deg; 60 WRs with >= 15 break routes.

| x | y | n | Spearman | 90% CI |
|---|---|---|---|---|
| in-game break entry speed | separation | 60 | +0.09 | -0.14 to +0.30 |
| in-game speed kept through the break | separation | 60 | +0.21 | +0.00 to +0.39 |
| Combine reversal entry speed | in-game break entry | 22 | +0.03 | -0.34 to +0.39 |
| Combine reversal entry speed | in-game retention | 22 | -0.26 | -0.63 to +0.16 |
| Combine composite | in-game break entry | 22 | +0.13 | -0.23 to +0.48 |

Fifth read: on the field it is the speed a receiver keeps through the cut, not the speed he brings
into it, that goes with separation. The Combine reversal entry speed does not show up as in-game entry
speed for the same players, so the +0.32 of the fourth pass is not "fast into the turn on Sundays".
Only 22 players have both, so the link stays open. Next: measure the Combine route drills (the
movement closest to a game cut) with this same break detector and compare retention to retention.

## Sixth pass: the same detector on the Combine route drills (analysis/06_combine_game_retention.py)
`break_features` from the fifth pass, unchanged, on the Combine route drills (slant, dagger, curl,
comeback, slot strike), drill-adjusted, per-player median over >= 3 reps. 321 drill cuts, 106 WRs,
median turn 124 deg; 51 WRs also have >= 15 break routes in 2023-25 games.

| x | y | n | Spearman | 90% CI |
|---|---|---|---|---|
| Combine route-cut entry speed | in-game break entry | 51 | +0.25 | +0.04 to +0.46 |
| Combine speed kept through the cut | in-game speed kept | 51 | +0.05 | -0.18 to +0.29 |
| Combine speed kept through the cut | separation | 51 | +0.06 | -0.19 to +0.29 |
| Combine route-cut entry speed | separation | 51 | +0.01 | -0.22 to +0.25 |
| in-game speed kept through the break | separation | 51 | +0.24 | +0.02 to +0.45 |

Sixth read: measured movement to movement, the Combine carries over the part that does not matter.
How fast a receiver arrives at a cut in shorts shows up again on Sundays (+0.25), but that entry speed
does not go with separation. How much speed he keeps through the cut is what goes with separation on
the field (+0.24 on these 51), and the Combine route drill does not see it (+0.05): an uncontested,
scripted cut is not the cut a receiver makes against a defender. Next: does keeping speed through
the cut depend on the defender's leverage at the break (nearest defender distance and side in the
game frames), which would explain why no drill reproduces it.

## Seventh pass: man against zone (analysis/07_coverage_split.py)
The fifth-pass break routes (no tracking re-read) joined to the defense's man/zone call and the
receiver's cushion at the snap (`player_play.csv`). Retention is route-adjusted; both retention and
separation have the cushion effect removed by a linear fit. 2,421 break routes with a call (783 man,
1,638 zone), 95 WRs. The gap interval resamples players, since routes from one receiver are clustered.

| level | coverage | n | Spearman (speed kept vs separation) | 90% CI |
|---|---|---|---|---|
| route | man | 783 | +0.08 | |
| route | zone | 1,638 | +0.02 | |
| route | man minus zone | 2,421 | +0.07 | +0.00 to +0.13 |
| player (>= 8 routes in that coverage) | man | 34 | +0.19 | -0.10 to +0.48 |
| player (>= 8 routes in that coverage) | zone | 57 | +0.09 | -0.13 to +0.30 |

Seventh read: the speed kept through the cut pays where a defender trails the receiver. Against man
it goes with separation four times as strongly as against zone, and the gap's interval sits at or
above zero; against zone, where the defender waits in an area, it is close to nothing. That is the
missing piece for the sixth read: the Combine cut is uncontested, so it cannot show the one skill
that only matters against a man defender. The effect per route is small and the player-level
intervals overlap, so this is a direction, not a ranking tool. Next: the defender's distance and
side at the break frame in man coverage, to see whether kept speed is the cause or the receivers
who keep speed are simply the ones defenders give room.

## Eighth pass: the defender at the break, man coverage (analysis/08_defender_at_break.py)
The man routes from the seventh pass, with the tracking re-read to find the same break frame as the
fifth pass and the defender there: distance, and leverage (cosine between the receiver's post-break
heading and the direction to the defender). A limit of the data showed up first: the tracking files
carry only Combine participants, one to three players per play, not all 22. The nearest tracked
defender at the break is a median 12.2 yd away, so most of the time he is not the man covering the
receiver. The defender tests use only the matchup routes, where a tracked defender is within 5 yd at
the break (188 routes, 64 WRs; 62% of those defenders sit on the side the receiver breaks to).
Separation at the throw comes from `player_play.csv` (nearest of all defenders), so "separation minus
distance at the break" mixes two different defenders and is reported only for completeness.

| subset | x | y | n | Spearman | 90% CI (players resampled) |
|---|---|---|---|---|---|
| all man | speed kept | separation over route | 783 | +0.07 | +0.02 to +0.12 |
| matchup | speed kept | defender distance at the break | 188 | +0.07 | -0.04 to +0.18 |
| matchup | speed kept | separation, defender distance held fixed | 188 | +0.15 | +0.06 to +0.24 |
| matchup | speed kept | separation over route | 188 | +0.15 | +0.07 to +0.24 |
| matchup, defender on the break side | speed kept | separation over route | 116 | +0.10 | -0.04 to +0.22 |
| matchup, defender trailing | speed kept | separation over route | 72 | +0.16 | +0.01 to +0.28 |

Eighth read: when the defender is really on the receiver, the speed kept through the cut goes with
separation twice as strongly as in man overall, and holding the defender's distance at the break
fixed does not change it. Receivers who keep speed are not the ones who were given room (+0.07, the
interval crosses zero). So the kept speed looks like the cause, not a side effect of a soft cushion.
It is strongest when the defender trails the cut, which is the situation the Combine drill never
creates. Small samples: a direction, not a ranking tool. Next: build the submission notebook around
the one chain the data supports (Combine entry repeats, kept speed does not, kept speed pays in man).

## Kaggle reproduction notebook (`kaggle/`)
`kaggle/combine-separation.ipynb` clones this repo on Kaggle, finds the attached competition data and
runs `analysis/01`-`08` in order, printing each summary table. Push with
`kaggle kernels push -p kaggle` (kernel `designmasterdms/does-the-combine-see-separation`, private until
the writeup is submitted, then public as the rules require).
