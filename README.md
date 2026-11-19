# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mizuseii/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mizuseii/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Write a program that reads the earn rate of every Grand Theft Auto VI trophy on Exophase as the house recorded it up to 2026-11-21 11:30 UTC and forecasts each trophy's rate at 2026-11-26 00:00 UTC, the percent of players who will have earned it by then, as five quantiles a trophy, for the lowest pinball loss on the logit scale — a Python 3.13 stdlib-only program that reads a make.fun case and writes five quantiles (5%, 25%, 50%, 75%, 95%) for each of the game's 55 trophies.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

- Exophase refreshes a game's counts on its own schedule, hours apart, not continuously. On launch night the page sat unchanged for hours (638 players tracked, every rate flat) and then refreshed in one jump to 3,868 players, with every rate but the 100% tutorial trophy falling sharply: the newly tracked players had earned almost nothing yet. Repeating the last rates therefore misses badly whenever a refresh lands between the checkpoint and the target — and over the final's 6,510-minute horizon several refreshes are near-certain.
- Project the tracked-player count at the target: a log-linear least-squares fit of ln(N) on hours since the first read when the history grows; when the history is flat, a refresh prior (about one refresh per 4.7 hours, each multiplying N by ~6x early on).
- Per trophy, estimate the newcomer earn slope g = dE/dN (E = players holding the trophy) by regressing earned counts on tracked players, shrunk toward the prior g = 0.07 * r/100 — new players earn a trophy at ~7% of the incumbents' rate — except a trophy at 100%, which newcomers get immediately (g = 1). Existing players also progress a little (up to 5%, scaled by how much of a 4.5-day window the horizon covers). The median is then 100 * E'/N' at the target.
- Spread on the logit scale is sqrt(0.25 * d_days + 0.25), plus 0.15 for the platinum; quantiles are inv_logit(logit(q50) + z * s).
- Deterministic, no network, no randomness; ~0.05 s per case.

## Validation (practice cases p00/p01, scored against the reconstructed refreshed answer)

| | pinball loss | typical miss | 90% coverage |
|---|---|---|---|
| This model | **0.355** | 0.18 pp | 0.982 |
| House persistence baseline | 1.40 | 0.68 pp | 0.000 |
| No-refresh forecast (last rates) | 1.485 | 0.68 pp | 0.018 |

On the 2 practice cases all outputs are valid (55 rows, ordered quantiles, ~2.8 KB each) and byte-identical across repeated runs. A synthetic final-like case (growing N history, 6,510-minute horizon) gives sane medians: falling rates for ordinary trophies, the 100% trophy stable at 99.99%.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
