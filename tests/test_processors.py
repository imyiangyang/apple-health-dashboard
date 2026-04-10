import pandas as pd
import pytest
from processors import (
    load_csv,
    deduplicate_activity_sources,
    aggregate_daily,
    add_rolling_averages,
    scale_records,
    reconstruct_sleep_sessions,
    group_running_sessions,
    group_swimming_sessions,
    compute_overview_stats,
    compute_yearly_summary,
    compute_data_density,
    HEART_RATE,
    STEPS,
    WATCH_SOURCES,
)


def test_processors_imports():
    import processors

    assert hasattr(processors, "HEART_RATE")
    assert hasattr(processors, "WATCH_SOURCES")


class TestLoadCsv:
    def test_returns_dataframe(self):
        df = load_csv("1_Heart_Metrics.csv")
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = load_csv("1_Heart_Metrics.csv")
        assert {"type", "value", "unit", "startdate", "sourcename", "date"}.issubset(
            df.columns
        )

    def test_startdate_is_datetime(self):
        df = load_csv("1_Heart_Metrics.csv")
        assert pd.api.types.is_datetime64_any_dtype(df["startdate"])

    def test_date_is_string_ymd(self):
        df = load_csv("1_Heart_Metrics.csv")
        sample = df["date"].iloc[0]
        assert len(sample) == 10 and sample[4] == "-" and sample[7] == "-"

    def test_no_bom_in_columns(self):
        df = load_csv("1_Heart_Metrics.csv")
        assert "\ufeff" not in df.columns[0]

    def test_value_numeric_for_heart(self):
        df = load_csv("1_Heart_Metrics.csv")
        assert pd.api.types.is_float_dtype(df["value"])

    def test_sleep_value_stays_string(self):
        df = load_csv("4_Sleep_Analysis.csv")
        assert df["value"].dtype == object
        assert "HKCategoryValueSleepAnalysis" in df["value"].iloc[0]


class TestDeduplication:
    def test_drops_iphone_when_watch_present(self, sample_steps_df):
        result = deduplicate_activity_sources(sample_steps_df)
        jan1 = result[result["date"] == "2024-01-01"]
        assert set(jan1["sourcename"].unique()).issubset(WATCH_SOURCES)

    def test_keeps_iphone_when_no_watch(self, sample_steps_df):
        result = deduplicate_activity_sources(sample_steps_df)
        jan2 = result[result["date"] == "2024-01-02"]
        assert len(jan2) == 2
        assert all(jan2["sourcename"] == "imyiangyang iPhone")

    def test_watch_row_count_after_dedup(self, sample_steps_df):
        result = deduplicate_activity_sources(sample_steps_df)
        jan1 = result[result["date"] == "2024-01-01"]
        assert len(jan1) == 2

    def test_other_metrics_unchanged(self):
        df = pd.DataFrame(
            {
                "type": [HEART_RATE] * 2,
                "value": [72.0, 80.0],
                "unit": ["count/min"] * 2,
                "startdate": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"]),
                "sourcename": ["imyiangyang iPhone", "Yang的Apple Watch"],
                "date": ["2024-01-01", "2024-01-01"],
            }
        )
        result = deduplicate_activity_sources(df)
        assert len(result) == 2


class TestDailyAggregation:
    def test_one_row_per_day(self, sample_hr_df):
        result = aggregate_daily(sample_hr_df, HEART_RATE)
        assert len(result) == 2

    def test_correct_mean(self, sample_hr_df):
        result = aggregate_daily(sample_hr_df, HEART_RATE)
        jan1 = result[result["date"] == "2024-01-01"].iloc[0]
        assert abs(jan1["mean"] - (72 + 80 + 75) / 3) < 0.01

    def test_correct_min_max(self, sample_hr_df):
        result = aggregate_daily(sample_hr_df, HEART_RATE)
        jan1 = result[result["date"] == "2024-01-01"].iloc[0]
        assert jan1["min"] == 72.0
        assert jan1["max"] == 80.0

    def test_empty_for_unknown_metric(self, sample_hr_df):
        result = aggregate_daily(sample_hr_df, "HKQuantityTypeIdentifierNonExistent")
        assert len(result) == 0
        assert "date" in result.columns

    def test_rolling_averages_added(self, sample_hr_df):
        daily = aggregate_daily(sample_hr_df, HEART_RATE)
        result = add_rolling_averages(daily)
        assert "ma7" in result.columns
        assert "ma30" in result.columns

    def test_rolling_ma7_with_three_points(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "mean": [10.0, 20.0, 30.0],
                "min": [10.0, 20.0, 30.0],
                "max": [10.0, 20.0, 30.0],
                "count": [1, 1, 1],
            }
        )
        result = add_rolling_averages(df)
        assert abs(result.iloc[0]["ma7"] - 10.0) < 0.01
        assert abs(result.iloc[1]["ma7"] - 15.0) < 0.01
        assert abs(result.iloc[2]["ma7"] - 20.0) < 0.01

    def test_scale_records(self):
        records = [
            {
                "date": "2024-01-01",
                "mean": 0.97,
                "min": 0.95,
                "max": 0.99,
                "ma7": 0.97,
                "ma30": 0.97,
            }
        ]
        scaled = scale_records(records, 100.0)
        assert abs(scaled[0]["mean"] - 97.0) < 0.01
        assert abs(scaled[0]["min"] - 95.0) < 0.01


class TestSleepReconstruction:
    def test_watch_session_detected(self, sample_sleep_df):
        sessions = reconstruct_sleep_sessions(sample_sleep_df)
        complete = [s for s in sessions if s["total_hours"] >= 1]
        assert len(complete) >= 1

    def test_required_fields(self, sample_sleep_df):
        sessions = reconstruct_sleep_sessions(sample_sleep_df)
        for s in sessions:
            for field in [
                "date",
                "bedtime",
                "wake_time",
                "total_hours",
                "sleep_hours",
                "efficiency",
                "deep_pct",
                "rem_pct",
                "core_pct",
                "awake_pct",
                "has_stages",
                "weekday",
            ]:
                assert field in s, f"Missing field: {field}"

    def test_night1_spans_8_hours(self, sample_sleep_df):
        sessions = reconstruct_sleep_sessions(sample_sleep_df)
        night1 = [s for s in sessions if s["date"] == "2024-01-02"]
        assert len(night1) == 1
        assert 5 <= night1[0]["total_hours"] <= 10

    def test_watch_session_has_stages(self, sample_sleep_df):
        sessions = reconstruct_sleep_sessions(sample_sleep_df)
        night1 = [s for s in sessions if s["date"] == "2024-01-02"]
        assert night1[0]["has_stages"] is True

    def test_clock_sleep_no_stages(self, sample_clock_sleep_df):
        sessions = reconstruct_sleep_sessions(sample_clock_sleep_df)
        assert len(sessions) == 1
        assert sessions[0]["has_stages"] is False
        assert abs(sessions[0]["total_hours"] - 7.0) < 0.1


class TestRunningGroups:
    def test_two_sessions_detected(self, sample_running_df):
        sessions = group_running_sessions(sample_running_df)
        assert len(sessions) == 2

    def test_required_fields(self, sample_running_df):
        sessions = group_running_sessions(sample_running_df)
        for s in sessions:
            for f in [
                "date",
                "time",
                "duration_min",
                "avg_speed_mps",
                "pace_min_per_km",
                "avg_stride_m",
                "avg_power_w",
                "avg_gct_ms",
                "avg_vosc_cm",
            ]:
                assert f in s

    def test_pace_calculation(self, sample_running_df):
        sessions = group_running_sessions(sample_running_df)
        s1 = sessions[0]
        expected = round(1000 / (s1["avg_speed_mps"] * 60), 2)
        assert abs(s1["pace_min_per_km"] - expected) < 0.05

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(
            columns=["type", "value", "startdate", "sourcename", "date"]
        )
        assert group_running_sessions(empty) == []


class TestSwimmingGroups:
    def test_two_sessions(self, sample_swim_df):
        sessions = group_swimming_sessions(sample_swim_df)
        assert len(sessions) == 2

    def test_distance_summed(self, sample_swim_df):
        sessions = group_swimming_sessions(sample_swim_df)
        for s in sessions:
            assert s["total_distance_m"] == 64.0

    def test_required_fields(self, sample_swim_df):
        sessions = group_swimming_sessions(sample_swim_df)
        for s in sessions:
            for f in [
                "date",
                "time",
                "duration_min",
                "total_distance_m",
                "total_strokes",
                "num_laps",
                "pace_min_per_100m",
            ]:
                assert f in s


class TestOverviewStats:
    def test_structure(self):
        data = {
            "resting_hr": [
                {
                    "date": "2024-01-01",
                    "mean": 62.0,
                    "ma7": 62.0,
                    "ma30": 62.0,
                    "min": 62.0,
                    "max": 62.0,
                    "count": 1,
                }
            ],
            "steps": [
                {
                    "date": "2024-01-01",
                    "mean": 8000.0,
                    "ma7": 8000.0,
                    "ma30": 8000.0,
                    "min": 8000.0,
                    "max": 8000.0,
                    "count": 1,
                }
            ],
            "sleep": [
                {
                    "date": "2024-01-02",
                    "sleep_hours": 7.5,
                    "total_hours": 8.0,
                    "efficiency": 93.0,
                    "deep_pct": 18.0,
                    "rem_pct": 22.0,
                    "core_pct": 55.0,
                    "awake_pct": 5.0,
                    "has_stages": True,
                    "weekday": 1,
                    "bedtime": "23:00",
                    "wake_time": "07:00",
                }
            ],
            "spo2": [
                {
                    "date": "2024-01-01",
                    "mean": 97.5,
                    "ma7": 97.5,
                    "ma30": 97.5,
                    "min": 97.0,
                    "max": 98.0,
                    "count": 1,
                }
            ],
            "walking_speed": [
                {
                    "date": "2024-01-01",
                    "mean": 4.8,
                    "ma7": 4.8,
                    "ma30": 4.8,
                    "min": 4.5,
                    "max": 5.0,
                    "count": 1,
                }
            ],
            "running_sessions": [],
            "swimming_sessions": [],
            "bmi": [],
        }
        stats = compute_overview_stats(data)
        for key in [
            "latest_resting_hr",
            "avg_steps_30d",
            "avg_sleep_hours_30d",
            "avg_spo2_30d",
            "latest_walking_speed",
            "running_sessions_90d",
            "total_swimming_sessions",
        ]:
            assert key in stats, f"Missing key: {key}"

    def test_yearly_summary_one_row_per_year(self):
        rhr = [
            {"date": "2023-06-01", "mean": 60.0},
            {"date": "2024-01-01", "mean": 58.0},
        ]
        steps = [
            {"date": "2023-06-01", "mean": 9000.0},
            {"date": "2024-01-01", "mean": 10000.0},
        ]
        summary = compute_yearly_summary(rhr, steps, [], [], [], [])
        assert len(summary) == 2
        assert {r["year"] for r in summary} == {2023, 2024}

    def test_data_density(self):
        metrics = {
            "hr": [{"date": "2024-01-01"}, {"date": "2024-01-02"}],
            "steps": [{"date": "2024-01-01"}],
        }
        density = compute_data_density(metrics)
        by_date = {d["date"]: d["count"] for d in density}
        assert by_date["2024-01-01"] == 2
        assert by_date["2024-01-02"] == 1
