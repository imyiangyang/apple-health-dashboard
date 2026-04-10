import pandas as pd
import pytest


@pytest.fixture
def sample_hr_df():
    return pd.DataFrame(
        {
            "type": ["HKQuantityTypeIdentifierHeartRate"] * 6,
            "value": [72.0, 80.0, 75.0, 68.0, 90.0, 85.0],
            "unit": ["count/min"] * 6,
            "startdate": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 10:00:00",
                    "2024-01-01 14:00:00",
                    "2024-01-02 09:00:00",
                    "2024-01-02 11:00:00",
                    "2024-01-02 15:00:00",
                ]
            ),
            "sourcename": ["Yang的Apple Watch"] * 6,
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ],
        }
    )


@pytest.fixture
def sample_steps_df():
    return pd.DataFrame(
        {
            "type": ["HKQuantityTypeIdentifierStepCount"] * 6,
            "value": [1000.0, 500.0, 800.0, 400.0, 1200.0, 100.0],
            "unit": ["count"] * 6,
            "startdate": pd.to_datetime(
                [
                    "2024-01-01 09:00:00",
                    "2024-01-01 09:05:00",
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:05:00",
                    "2024-01-02 09:00:00",
                    "2024-01-02 10:00:00",
                ]
            ),
            "sourcename": [
                "Yang的Apple Watch",
                "imyiangyang iPhone",
                "Yang的Apple Watch",
                "imyiangyang iPhone",
                "imyiangyang iPhone",
                "imyiangyang iPhone",
            ],
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
            ],
        }
    )


@pytest.fixture
def sample_sleep_df():
    return pd.DataFrame(
        {
            "type": ["HKCategoryTypeIdentifierSleepAnalysis"] * 10,
            "value": [
                "HKCategoryValueSleepAnalysisInBed",
                "HKCategoryValueSleepAnalysisAsleepCore",
                "HKCategoryValueSleepAnalysisAsleepDeep",
                "HKCategoryValueSleepAnalysisAsleepREM",
                "HKCategoryValueSleepAnalysisAsleepCore",
                "HKCategoryValueSleepAnalysisAsleepCore",
                "HKCategoryValueSleepAnalysisAsleepREM",
                "HKCategoryValueSleepAnalysisAsleepDeep",
                "HKCategoryValueSleepAnalysisAwake",
                "HKCategoryValueSleepAnalysisAsleepCore",
            ],
            "unit": [""] * 10,
            "startdate": pd.to_datetime(
                [
                    "2024-01-01 22:00:00",
                    "2024-01-01 22:30:00",
                    "2024-01-01 23:30:00",
                    "2024-01-02 00:30:00",
                    "2024-01-02 01:30:00",
                    "2024-01-02 02:30:00",
                    "2024-01-02 03:30:00",
                    "2024-01-02 04:30:00",
                    "2024-01-02 06:00:00",
                    "2024-01-02 23:00:00",
                ]
            ),
            "sourcename": ["Yang的Apple Watch"] * 10,
            "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 6,
        }
    )


@pytest.fixture
def sample_clock_sleep_df():
    return pd.DataFrame(
        {
            "type": ["HKCategoryTypeIdentifierSleepAnalysis"] * 2,
            "value": ["HKCategoryValueSleepAnalysisInBed"] * 2,
            "unit": [""] * 2,
            "startdate": pd.to_datetime(
                [
                    "2019-01-10 01:30:00",
                    "2019-01-10 08:30:00",
                ]
            ),
            "sourcename": ["Clock"] * 2,
            "date": ["2019-01-10", "2019-01-10"],
        }
    )


@pytest.fixture
def sample_running_df():
    times = pd.to_datetime(
        [
            "2024-01-10 07:00:00",
            "2024-01-10 07:00:30",
            "2024-01-10 07:01:00",
            "2024-01-10 08:00:00",
            "2024-01-10 08:00:30",
            "2024-01-10 08:01:00",
        ]
    )
    return pd.DataFrame(
        {
            "type": ["HKQuantityTypeIdentifierRunningSpeed"] * 6,
            "value": [3.0, 3.5, 3.2, 4.0, 4.2, 4.1],
            "unit": ["m/s"] * 6,
            "startdate": times,
            "sourcename": ["Yang的Apple Watch"] * 6,
            "date": ["2024-01-10"] * 6,
        }
    )


@pytest.fixture
def sample_swim_df():
    return pd.DataFrame(
        {
            "type": ["HKQuantityTypeIdentifierDistanceSwimming"] * 4,
            "value": [32.0, 32.0, 32.0, 32.0],
            "unit": ["m"] * 4,
            "startdate": pd.to_datetime(
                [
                    "2025-07-12 19:56:00",
                    "2025-07-12 19:57:00",
                    "2025-07-13 18:00:00",
                    "2025-07-13 18:01:00",
                ]
            ),
            "sourcename": ["Yang的Apple Watch"] * 4,
            "date": ["2025-07-12", "2025-07-12", "2025-07-13", "2025-07-13"],
        }
    )
