"""Apple Health data processing functions.

All functions are pure and testable. No side effects.
Units returned (after conversion where noted):
  - walking_speed: km/h  (source: m/s × 3.6)
  - step_length:   cm    (source: m × 100)
  - double_support: %    (source: fraction × 100)
  - asymmetry:     %     (source: fraction × 100)
  - spo2:          %     (source: fraction × 100)
  - running_speed: m/s   (raw; pace computed separately as min/km)
  - gct:           ms    (raw from Apple Health)
  - vosc:          cm    (raw from Apple Health)
"""

from datetime import timedelta, date as date_type
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent

# ── HKType constants ──────────────────────────────────────────────────────────
HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
RESTING_HR = "HKQuantityTypeIdentifierRestingHeartRate"
HRV_SDNN = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
HR_RECOVERY = "HKQuantityTypeIdentifierHeartRateRecoveryOneMinute"
WALKING_HR = "HKQuantityTypeIdentifierWalkingHeartRateAverage"

BODY_MASS = "HKQuantityTypeIdentifierBodyMass"
BMI = "HKQuantityTypeIdentifierBodyMassIndex"
BODY_FAT = "HKQuantityTypeIdentifierBodyFatPercentage"
LEAN_MASS = "HKQuantityTypeIdentifierLeanBodyMass"

STEPS = "HKQuantityTypeIdentifierStepCount"
ACTIVE_ENERGY = "HKQuantityTypeIdentifierActiveEnergyBurned"
BASAL_ENERGY = "HKQuantityTypeIdentifierBasalEnergyBurned"
DISTANCE_WR = "HKQuantityTypeIdentifierDistanceWalkingRunning"
FLIGHTS = "HKQuantityTypeIdentifierFlightsClimbed"
CYCLING = "HKQuantityTypeIdentifierDistanceCycling"
SIX_MIN_WALK = "HKQuantityTypeIdentifierSixMinuteWalkTestDistance"

SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
SLEEP_INBED = "HKCategoryValueSleepAnalysisInBed"
SLEEP_CORE = "HKCategoryValueSleepAnalysisAsleepCore"
SLEEP_DEEP = "HKCategoryValueSleepAnalysisAsleepDeep"
SLEEP_REM = "HKCategoryValueSleepAnalysisAsleepREM"
SLEEP_AWAKE = "HKCategoryValueSleepAnalysisAwake"
SLEEP_UNSPECIFIED = "HKCategoryValueSleepAnalysisAsleepUnspecified"

WALKING_SPEED = "HKQuantityTypeIdentifierWalkingSpeed"
STEP_LENGTH = "HKQuantityTypeIdentifierWalkingStepLength"
DOUBLE_SUPPORT = "HKQuantityTypeIdentifierWalkingDoubleSupportPercentage"
ASYMMETRY = "HKQuantityTypeIdentifierWalkingAsymmetryPercentage"
STEADINESS = "HKQuantityTypeIdentifierAppleWalkingSteadiness"

SPO2 = "HKQuantityTypeIdentifierOxygenSaturation"
RESP_RATE = "HKQuantityTypeIdentifierRespiratoryRate"

RUNNING_STRIDE = "HKQuantityTypeIdentifierRunningStrideLength"
RUNNING_SPEED = "HKQuantityTypeIdentifierRunningSpeed"
RUNNING_POWER = "HKQuantityTypeIdentifierRunningPower"
RUNNING_GCT = "HKQuantityTypeIdentifierRunningGroundContactTime"
RUNNING_VOSC = "HKQuantityTypeIdentifierRunningVerticalOscillation"

SWIM_DISTANCE = "HKQuantityTypeIdentifierDistanceSwimming"
SWIM_STROKES = "HKQuantityTypeIdentifierSwimmingStrokeCount"

WATCH_SOURCES = frozenset(["Yang的Apple Watch", "Yang's"])
DEDUP_METRICS = frozenset([STEPS, DISTANCE_WR])

# ── CSV Loading ───────────────────────────────────────────────────────────────


def load_csv(filename: str) -> pd.DataFrame:
    """Load an Apple Health CSV, handling BOM and parsing dates.

    Returns DataFrame with columns:
      type, value, unit, startdate (datetime), sourcename, date (str YYYY-MM-DD)

    For sleep CSVs the 'value' column is kept as string (category names).
    For all other CSVs 'value' is coerced to float64.
    """
    path = DATA_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # Parse startdate – stored with UTC offset already in CST (+0800); strip tz
    df["startdate"] = pd.to_datetime(df["startdate"], utc=False, format="mixed")
    if df["startdate"].dt.tz is not None:
        df["startdate"] = (
            df["startdate"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        )

    df["date"] = df["startdate"].dt.strftime("%Y-%m-%d")

    # Sleep CSVs have string category values; keep them as-is
    if not df.empty and df["type"].iloc[0] == SLEEP_TYPE:
        pass  # value stays as str
    else:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ── Deduplication ─────────────────────────────────────────────────────────────


def deduplicate_activity_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer Apple Watch over iPhone for StepCount and DistanceWalkingRunning.

    On days where Apple Watch data exists for a metric, iPhone rows for that
    (date, metric_type) group are dropped.  On days with only iPhone data the
    iPhone rows are kept unchanged.  All other metric types pass through.
    """
    mask = df["type"].isin(DEDUP_METRICS)
    to_dedup = df[mask].copy()
    others = df[~mask].copy()

    if to_dedup.empty:
        return df.copy()

    to_dedup["_has_watch"] = to_dedup.groupby(["date", "type"])["sourcename"].transform(
        lambda s: s.isin(WATCH_SOURCES).any()
    )
    kept = to_dedup[
        to_dedup["sourcename"].isin(WATCH_SOURCES) | ~to_dedup["_has_watch"]
    ].drop(columns=["_has_watch"])

    return pd.concat([others, kept], ignore_index=True)


# ── Daily Aggregation ─────────────────────────────────────────────────────────


def aggregate_daily(df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    """Aggregate one metric type to daily mean / min / max / count.

    Returns DataFrame sorted by date with columns:
      date (str), mean, min, max, count
    Returns empty DataFrame with those columns if metric_type not found.
    """
    sub = df[df["type"] == metric_type].copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "mean", "min", "max", "count"])

    daily = (
        sub.groupby("date")["value"]
        .agg(mean="mean", min="min", max="max", count="count")
        .reset_index()
    )
    daily["mean"] = daily["mean"].round(3)
    daily["min"] = daily["min"].round(3)
    daily["max"] = daily["max"].round(3)
    return daily.sort_values("date").reset_index(drop=True)


def add_rolling_averages(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add MA7 and MA30 columns to a daily DataFrame.

    Requires 'date' and 'mean' columns.  Uses min_periods=1 so the first
    days of the series always get a value.  NaN → None for JSON serialisation.
    """
    df = daily_df.copy().sort_values("date").reset_index(drop=True)
    df["ma7"] = df["mean"].rolling(window=7, min_periods=1).mean().round(3)
    df["ma30"] = df["mean"].rolling(window=30, min_periods=1).mean().round(3)
    # None serialises to JSON null; NaN does not
    df = df.where(df.notna(), other=None)
    return df


def scale_records(
    records: list[dict], factor: float, keys=("mean", "min", "max", "ma7", "ma30")
) -> list[dict]:
    """Multiply numeric fields in a list of dicts by factor (unit conversion)."""
    out = []
    for r in records:
        r2 = dict(r)
        for k in keys:
            if r2.get(k) is not None:
                r2[k] = round(r2[k] * factor, 3)
        out.append(r2)
    return out


# ── Sleep Session Reconstruction ──────────────────────────────────────────────

_SLEEP_STAGE_TYPES = frozenset(
    [SLEEP_CORE, SLEEP_DEEP, SLEEP_REM, SLEEP_AWAKE, SLEEP_UNSPECIFIED]
)

_VALUES_TO_STAGE = {
    SLEEP_INBED: "inbed",
    SLEEP_CORE: "core",
    SLEEP_DEEP: "deep",
    SLEEP_REM: "rem",
    SLEEP_AWAKE: "awake",
    SLEEP_UNSPECIFIED: "unspecified",
}


def reconstruct_sleep_sessions(sleep_df: pd.DataFrame) -> list[dict]:
    """Reconstruct nightly sleep sessions from Apple Health sleep records.

    Handles two source types:
    - Apple Watch: consecutive stage records (Core/Deep/REM/Awake) with
      small (minute-level) gaps.  Duration estimated as next_start − this_start.
    - Clock app:  sparse InBed records; typically two per night (bedtime +
      wake-time) separated by the full sleep span (hours).

    Returns list of session dicts sorted ascending by date (wake-up date).
    """
    if sleep_df.empty:
        return []

    watch = sleep_df[sleep_df["sourcename"].isin(WATCH_SOURCES)]
    clock = sleep_df[~sleep_df["sourcename"].isin(WATCH_SOURCES)]

    sessions = []
    if not watch.empty:
        sessions += _watch_sleep(watch)
    if not clock.empty:
        sessions += _clock_sleep(clock)

    return sorted(sessions, key=lambda s: s["date"])


def _watch_sleep(df: pd.DataFrame) -> list[dict]:
    """Process Apple Watch sleep records (consecutive stage approach)."""
    df = df.copy().sort_values("startdate").reset_index(drop=True)
    gaps = df["startdate"].diff().dt.total_seconds().fillna(0)
    df["session_id"] = (gaps > 2 * 3600).cumsum()  # >2 h gap = new session

    sessions = []
    for _, group in df.groupby("session_id"):
        group = group.sort_values("startdate").reset_index(drop=True)
        first_time = group["startdate"].iloc[0]
        last_time = group["startdate"].iloc[-1]
        span_hours = (last_time - first_time).total_seconds() / 3600

        if span_hours < 1:
            continue
        # Exclude daytime naps: start 06–18 h AND span < 3 h
        if 6 <= first_time.hour < 18 and span_hours < 3:
            continue

        has_stages = any(v in _SLEEP_STAGE_TYPES for v in group["value"])
        stage_min = {
            k: 0.0 for k in ["inbed", "core", "deep", "rem", "awake", "unspecified"]
        }

        for i in range(len(group) - 1):
            stage = _VALUES_TO_STAGE.get(group.iloc[i]["value"], "inbed")
            dur = (
                group.iloc[i + 1]["startdate"] - group.iloc[i]["startdate"]
            ).total_seconds() / 60
            stage_min[stage] += min(max(dur, 0), 120)  # cap at 2 h per segment

        total_sleep_min = sum(
            stage_min[k] for k in ["core", "deep", "rem", "unspecified"]
        )
        total_inbed_min = span_hours * 60
        if not has_stages:
            total_sleep_min = total_inbed_min  # Clock-style InBed only

        eff = (
            round(total_sleep_min / total_inbed_min * 100, 1)
            if total_inbed_min > 0
            else 0
        )

        def pct(m):
            return round(m / total_inbed_min * 100, 1) if total_inbed_min > 0 else 0

        sessions.append(
            {
                "date": last_time.strftime("%Y-%m-%d"),
                "bedtime": first_time.strftime("%H:%M"),
                "wake_time": last_time.strftime("%H:%M"),
                "total_hours": round(span_hours, 2),
                "sleep_hours": round(total_sleep_min / 60, 2),
                "efficiency": eff,
                "deep_pct": pct(stage_min["deep"]),
                "rem_pct": pct(stage_min["rem"]),
                "core_pct": pct(stage_min["core"]),
                "awake_pct": pct(stage_min["awake"]),
                "has_stages": has_stages,
                "weekday": last_time.weekday(),
            }
        )
    return sessions


def _clock_sleep(df: pd.DataFrame) -> list[dict]:
    """Process Clock-app sleep records (sparse InBed pairs)."""
    df = df.copy().sort_values("startdate").reset_index(drop=True)
    gaps = df["startdate"].diff().dt.total_seconds().fillna(0)
    # Clock records for the same night can be hours apart; use 14 h threshold
    df["night_id"] = (gaps > 14 * 3600).cumsum()

    sessions = []
    for _, group in df.groupby("night_id"):
        group = group.sort_values("startdate")
        bedtime = group["startdate"].iloc[0]
        wake_time = group["startdate"].iloc[-1]
        span_hours = (wake_time - bedtime).total_seconds() / 3600
        if span_hours < 1:
            continue
        sessions.append(
            {
                "date": wake_time.strftime("%Y-%m-%d"),
                "bedtime": bedtime.strftime("%H:%M"),
                "wake_time": wake_time.strftime("%H:%M"),
                "total_hours": round(span_hours, 2),
                "sleep_hours": round(span_hours, 2),
                "efficiency": 100.0,
                "deep_pct": 0.0,
                "rem_pct": 0.0,
                "core_pct": 0.0,
                "awake_pct": 0.0,
                "has_stages": False,
                "weekday": wake_time.weekday(),
            }
        )
    return sessions


# ── Running Session Grouping ──────────────────────────────────────────────────

_RUN_GAP_MIN = 5  # gap > 5 min between records → new session


def group_running_sessions(running_df: pd.DataFrame) -> list[dict]:
    """Group per-second running records into workout sessions.

    Detects new session when consecutive timestamps are > 5 minutes apart.
    Returns list of session dicts sorted by date.
    """
    if running_df.empty:
        return []

    df = running_df.copy().sort_values("startdate").reset_index(drop=True)
    gaps = df["startdate"].diff().dt.total_seconds().fillna(0)
    df["session_id"] = (gaps > _RUN_GAP_MIN * 60).cumsum()

    sessions = []
    for _, group in df.groupby("session_id"):
        group = group.sort_values("startdate")
        first_time = group["startdate"].iloc[0]
        last_time = group["startdate"].iloc[-1]
        duration_sec = (last_time - first_time).total_seconds()
        if duration_sec < 60:
            continue

        def avg(mtype):
            v = group[group["type"] == mtype]["value"].dropna()
            return round(float(v.mean()), 3) if len(v) > 0 else None

        avg_speed = avg(RUNNING_SPEED)  # m/s
        pace = (
            round(1000 / (avg_speed * 60), 2) if avg_speed and avg_speed > 0 else None
        )

        sessions.append(
            {
                "date": first_time.strftime("%Y-%m-%d"),
                "time": first_time.strftime("%H:%M"),
                "duration_min": round(duration_sec / 60, 1),
                "avg_speed_mps": avg_speed,
                "pace_min_per_km": pace,
                "avg_stride_m": avg(RUNNING_STRIDE),
                "avg_power_w": avg(RUNNING_POWER),
                "avg_gct_ms": avg(RUNNING_GCT),
                "avg_vosc_cm": avg(RUNNING_VOSC),
            }
        )
    return sorted(sessions, key=lambda s: s["date"])


# ── Swimming Session Grouping ─────────────────────────────────────────────────

_SWIM_GAP_MIN = 10
_SWIM_LAP_LENGTH = 32  # metres (inferred from data)


def group_swimming_sessions(swim_df: pd.DataFrame) -> list[dict]:
    """Group swimming records into sessions; compute distance, strokes, pace."""
    if swim_df.empty:
        return []

    df = swim_df.copy().sort_values("startdate").reset_index(drop=True)
    gaps = df["startdate"].diff().dt.total_seconds().fillna(0)
    df["session_id"] = (gaps > _SWIM_GAP_MIN * 60).cumsum()

    sessions = []
    for _, group in df.groupby("session_id"):
        group = group.sort_values("startdate")
        first_time = group["startdate"].iloc[0]
        last_time = group["startdate"].iloc[-1]
        duration_sec = (last_time - first_time).total_seconds()

        dist_vals = group[group["type"] == SWIM_DISTANCE]["value"].dropna()
        stroke_vals = group[group["type"] == SWIM_STROKES]["value"].dropna()

        total_dist = float(dist_vals.sum()) if len(dist_vals) > 0 else 0.0
        total_strokes = float(stroke_vals.sum()) if len(stroke_vals) > 0 else 0.0
        num_laps = round(total_dist / _SWIM_LAP_LENGTH, 1)

        pace = (
            round((duration_sec / 60) / total_dist * 100, 2)
            if total_dist > 0 and duration_sec > 0
            else None
        )

        sessions.append(
            {
                "date": first_time.strftime("%Y-%m-%d"),
                "time": first_time.strftime("%H:%M"),
                "duration_min": round(duration_sec / 60, 1),
                "total_distance_m": total_dist,
                "total_strokes": total_strokes,
                "num_laps": num_laps,
                "pace_min_per_100m": pace,
            }
        )
    return sorted(sessions, key=lambda s: s["date"])


# ── Overview Stats ────────────────────────────────────────────────────────────


def _last_n_avg(records: list[dict], key: str, n: int) -> Optional[float]:
    """Average `key` from records in the last n calendar days."""
    if not records:
        return None
    cutoff = (date_type.today() - timedelta(days=n)).isoformat()
    vals = [
        r[key]
        for r in records
        if r.get("date", "") >= cutoff and r.get(key) is not None
    ]
    return round(sum(vals) / len(vals), 2) if vals else None


def compute_overview_stats(data: dict) -> dict:
    """Compute metric card values for the Overview tab."""
    rhr = data.get("resting_hr", [])
    steps = data.get("steps", [])
    sleep = data.get("sleep", [])
    spo2 = data.get("spo2", [])
    wspeed = data.get("walking_speed", [])
    run = data.get("running_sessions", [])
    swim = data.get("swimming_sessions", [])
    bmi = data.get("bmi", [])

    latest_rhr = rhr[-1]["mean"] if rhr else None
    avg_rhr_30 = _last_n_avg(rhr, "mean", 30)
    # Trend: compare latest 30d avg vs prior 30d avg
    cutoff_60 = (date_type.today() - timedelta(days=60)).isoformat()
    cutoff_30 = (date_type.today() - timedelta(days=30)).isoformat()
    prior = [r for r in rhr if cutoff_60 <= r.get("date", "") < cutoff_30]
    avg_rhr_prior = (
        round(sum(r["mean"] for r in prior) / len(prior), 1) if prior else None
    )
    trend = "→"
    if avg_rhr_30 and avg_rhr_prior:
        diff = avg_rhr_30 - avg_rhr_prior
        trend = "↑" if diff > 2 else ("↓" if diff < -2 else "→")

    cutoff_90 = (date_type.today() - timedelta(days=90)).isoformat()
    run_90 = len([s for s in run if s.get("date", "") >= cutoff_90])

    avg_spo2 = _last_n_avg(spo2, "mean", 30)

    return {
        "latest_resting_hr": latest_rhr,
        "trend_resting_hr": trend,
        "latest_bmi": bmi[-1]["mean"] if bmi else None,
        "avg_steps_30d": _last_n_avg(steps, "mean", 30),
        "avg_sleep_hours_30d": _last_n_avg(sleep, "sleep_hours", 30),
        "avg_spo2_30d": round(avg_spo2, 1) if avg_spo2 else None,
        "latest_walking_speed": wspeed[-1]["mean"] if wspeed else None,
        "running_sessions_90d": run_90,
        "total_swimming_sessions": len(swim),
    }


def compute_yearly_summary(
    resting_hr: list[dict],
    steps: list[dict],
    sleep: list[dict],
    bmi: list[dict],
    running_sessions: list[dict],
    swimming_sessions: list[dict],
) -> list[dict]:
    """One-row-per-year summary for the Overview table."""
    all_years = set()
    for records in [resting_hr, steps, sleep]:
        all_years.update(r["date"][:4] for r in records if r.get("date"))
    if not all_years:
        return []

    def yr_avg(records, key, year):
        vals = [
            r[key]
            for r in records
            if r.get("date", "").startswith(year) and r.get(key) is not None
        ]
        return round(sum(vals) / len(vals), 1) if vals else None

    rows = []
    for year in sorted(all_years):
        rows.append(
            {
                "year": int(year),
                "avg_resting_hr": yr_avg(resting_hr, "mean", year),
                "avg_steps": yr_avg(steps, "mean", year),
                "avg_sleep_hours": yr_avg(sleep, "sleep_hours", year),
                "avg_bmi": yr_avg(bmi, "mean", year),
                "running_sessions": len(
                    [s for s in running_sessions if s.get("date", "").startswith(year)]
                ),
                "swimming_sessions": len(
                    [s for s in swimming_sessions if s.get("date", "").startswith(year)]
                ),
            }
        )
    return rows


def compute_data_density(all_daily_metrics: dict) -> list[dict]:
    """Count how many metric types have data on each date (for heatmap)."""
    date_counts: dict[str, int] = {}
    for _, records in all_daily_metrics.items():
        for r in records:
            d = r.get("date")
            if d:
                date_counts[d] = date_counts.get(d, 0) + 1
    return [{"date": d, "count": c} for d, c in sorted(date_counts.items())]
