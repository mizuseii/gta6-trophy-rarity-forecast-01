"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles (5/25/50/75/95%).

The key fact about the data: Exophase refreshes a game's counts on its own
schedule, hours apart, not continuously. On launch night the page sat
unchanged for hours (638 players tracked, every rate flat) and then refreshed
in one jump to 3,868 players, with every rate but the 100% tutorial trophy
falling sharply: the newly tracked players had earned almost nothing yet.
A forecast that just repeats the last rates therefore misses badly whenever
a refresh lands between the checkpoint and the target -- and over the
final's 6,510-minute horizon several refreshes are near-certain.

Model, per case and per trophy (r = last non-null rate, N = players tracked):
  1. Project the tracked-player count N' at the target. A log-linear
     least-squares fit of ln(N) on hours since the first read gives the
     growth rate when the history actually grows; when the history is flat
     (slope <= 0) fall back on the refresh prior: about one refresh per
     4.7 hours, each multiplying N by ~6x early on, so
     N' = N_last * (1 + min(1, H/4.7) * 5).
  2. Per-trophy earn slope g_i = dE_i/dN, with E_i = rate/100 * N the number
     of tracked players holding the trophy. From history, regress E_i on N;
     with a flat history use the newcomer prior g_i = 0.07 * r_i/100 -- new
     players earn a trophy at ~7% of the incumbents' rate -- except a trophy
     at >= 99.99% (the tutorial trophy newcomers get immediately), where
     g_i = 1. History slopes shrink toward the prior with weight
     w = dN_total / (dN_total + 2000). A long horizon adds a maturity
     uplift: newcomers have had longer to play, so g_i grows slowly with
     ln(1 + H_days).
  3. Existing players also progress a little: E_i grows by up to 5%,
     scaled by (1 - r_i/100) and by how much of a 4.5-day window the
     horizon covers.
  4. Median: E_i' = E_i + progression + g_i * (N' - N_last), and
     q50 = 100 * E_i' / N', clipped to [0.01, 99.99].
  5. Spread on the logit scale: s_i = sqrt(0.25 * d_days + 0.25), plus
     0.15 for the platinum (its rate is a sum over the whole game, so its
     refresh move is wider). Quantiles are inv_logit(logit(q50) + z * s_i).

Standard library only, deterministic, no network, no randomness.
Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _parse_time(s):
    """Parse an ISO 8601 timestamp (with Z or an offset) to an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _logit(p):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def _inv_logit(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _round4(x):
    """Round to 4 decimals, half up (JS Math.round style for positives)."""
    return float(Decimal(repr(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _clip_rate(x):
    return min(max(x, 0.01), 99.99)


def _last_rate(rates):
    """Last non-null rate in a trophy's history column, else None."""
    for v in reversed(rates):
        if v is not None:
            return float(v)
    return None


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_min = float(case.get("horizon_minutes", 0))
    horizon_h = horizon_min / 60.0
    horizon_d = horizon_min / 1440.0
    n_tr = len(trophies)

    # --- tracked-player counts and read times -------------------------------
    counts = []
    times = []
    for read in history:
        n = read.get("players_tracked")
        if n is None:
            continue
        counts.append(float(n))
        try:
            times.append(_parse_time(read["at"]))
        except Exception:
            times.append(None)
    if not counts:
        counts = [1.0]
        times = [None]
    n_last = counts[-1]
    n_first = counts[0]

    # Hours since the first read, for the log-linear N fit.
    t0 = next((t for t in times if t is not None), None)
    hours = []
    for t in times:
        if t is None or t0 is None:
            hours.append(0.0)
        else:
            hours.append((t - t0).total_seconds() / 3600.0)

    # --- project N' at the target -------------------------------------------
    n_proj = None
    if len(counts) >= 2:
        xs = hours
        ys = [math.log(max(c, 1.0)) for c in counts]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            if slope > 0:
                g = min(slope, 0.04)
                ratio = math.exp(g * horizon_h)
                ratio = min(ratio, 15.0)
                n_proj = n_last * ratio
    if n_proj is None:
        # Flat history (or no usable slope): the refresh prior. About one
        # refresh per 4.7 h, each multiplying N by ~6x early on.
        frac = min(1.0, horizon_h / 4.7) if horizon_h > 0 else 0.0
        n_proj = n_last * (1.0 + frac * 5.0)
    if n_proj < n_last:
        n_proj = n_last
    d_n = n_proj - n_last

    # --- per-trophy last rates and earn counts ------------------------------
    # rates_by_trophy[i] = list of (count, rate) pairs over the reads.
    series = [[] for _ in range(n_tr)]
    for read in history:
        n = read.get("players_tracked")
        rates = read.get("rates") or []
        if n is None:
            continue
        for i in range(n_tr):
            if i < len(rates) and rates[i] is not None:
                series[i].append((float(n), float(rates[i])))

    total_dn = max(n_last - n_first, 0.0)
    w_hist = total_dn / (total_dn + 2000.0) if total_dn > 0 else 0.0
    maturity = 1.0 + 0.08 * math.log1p(horizon_d)
    prog_factor = 0.05 * min(horizon_d / 4.5, 1.0) if horizon_d > 0 else 0.0

    # Spread on the logit scale; wider for the platinum.
    spread_base = math.sqrt(0.25 * horizon_d + 0.25)

    rows = []
    for i, tr in enumerate(trophies):
        r = _last_rate([rate for _, rate in series[i]])
        if r is None:
            r = 1.0
        r = min(max(r, 0.0), 100.0)
        tier = str(tr.get("tier", "")).lower()

        # Prior newcomer earn slope: new players earn at ~7% of the
        # incumbents' rate; the 100% trophy is earned immediately (g = 1).
        if r >= 99.99:
            g_prior = 1.0
        else:
            g_prior = 0.07 * r / 100.0

        # History slope: regress earned counts E on tracked players N.
        g_hist = None
        pts = series[i]
        if len(pts) >= 2:
            ns = [n for n, _ in pts]
            es = [rate / 100.0 * n for n, rate in pts]
            mn = sum(ns) / len(ns)
            me = sum(es) / len(es)
            var_n = sum((n - mn) ** 2 for n in ns)
            if var_n > 0:
                g_hist = sum((n - mn) * (e - me) for n, e in zip(ns, es)) / var_n

        if g_hist is None:
            g = g_prior
        else:
            g = w_hist * g_hist + (1.0 - w_hist) * g_prior
        g = min(max(g, 0.0), 1.0)
        g *= maturity
        # The maturity uplift must not turn a bounded earn slope into an
        # unbounded one: a trophy cannot be earned by more newcomers than
        # arrive. Re-clip after the uplift (the 100% trophy's g = 1 stays
        # effectively capped; q50's own clip is the final backstop).
        g = min(max(g, 0.0), 1.15)

        # Existing-player progression, then newcomers' earns.
        e_last = r / 100.0 * n_last
        e_new = e_last
        e_new += e_last * prog_factor * (1.0 - r / 100.0)
        e_new += g * d_n
        e_new = max(e_new, 0.0)

        q50 = _clip_rate(100.0 * e_new / n_proj) if n_proj > 0 else _clip_rate(r)
        s = spread_base + (0.15 if tier == "platinum" else 0.0)
        l50 = _logit(q50 / 100.0)
        qs = []
        for z in Z:
            q = 100.0 * _inv_logit(l50 + z * s)
            qs.append(_clip_rate(_round4(q)))
        # Keep the row non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})

    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        with open(os.path.join(in_dir, fn), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        out_path = os.path.join(out_dir, case["id"] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
