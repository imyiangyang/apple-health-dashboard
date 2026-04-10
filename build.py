from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from processors import (
    ACTIVE_ENERGY,
    ASYMMETRY,
    BASAL_ENERGY,
    BMI,
    BODY_FAT,
    BODY_MASS,
    CYCLING,
    DISTANCE_WR,
    DOUBLE_SUPPORT,
    FLIGHTS,
    HEART_RATE,
    HR_RECOVERY,
    HRV_SDNN,
    LEAN_MASS,
    RESTING_HR,
    RESP_RATE,
    RUNNING_GCT,
    RUNNING_POWER,
    RUNNING_SPEED,
    RUNNING_STRIDE,
    RUNNING_VOSC,
    SIX_MIN_WALK,
    SPO2,
    STEADINESS,
    STEPS,
    SWIM_DISTANCE,
    SWIM_STROKES,
    WALKING_HR,
    WALKING_SPEED,
    STEP_LENGTH,
    add_rolling_averages,
    aggregate_daily,
    compute_data_density,
    compute_overview_stats,
    compute_yearly_summary,
    deduplicate_activity_sources,
    group_running_sessions,
    group_swimming_sessions,
    load_csv,
    reconstruct_sleep_sessions,
    scale_records,
)

CUTOFF_DATE = "2021-04-01"


def _filter_recent(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns and not df.empty:
        df = df[df["date"] >= CUTOFF_DATE].copy()
    return df


def _remove_outliers(
    records: list[dict], key: str = "mean", factor: float = 2.0
) -> list[dict]:
    vals = [r[key] for r in records if r.get(key) is not None]
    if len(vals) < 10:
        return records
    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    q1 = vals_sorted[n // 4]
    q3 = vals_sorted[3 * n // 4]
    iqr = q3 - q1
    if iqr == 0:
        return records
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return [r for r in records if r.get(key) is None or lower <= r[key] <= upper]


METRIC_INFO = {
    "resting_hr": {
        "name": "静息心率",
        "unit": "bpm",
        "desc": "安静状态下的心率，反映心脏基础功能。正常范围60-100 bpm，长期运动者可能更低。",
        "good": "lower",
    },
    "hrv_sdnn": {
        "name": "心率变异性 (HRV)",
        "unit": "ms",
        "desc": "心跳间隔的变异程度，反映自主神经系统平衡。数值越高通常代表恢复能力越好。",
        "good": "higher",
    },
    "hr_recovery": {
        "name": "心率恢复",
        "unit": "bpm",
        "desc": "运动后1分钟心率下降值，反映心血管适应能力。>12 bpm为正常。",
        "good": "higher",
    },
    "walking_hr": {
        "name": "步行心率",
        "unit": "bpm",
        "desc": "日常步行时的平均心率，反映有氧效率。同等活动量下数值越低越好。",
        "good": "lower",
    },
    "heart_rate": {
        "name": "全天心率",
        "unit": "bpm",
        "desc": "全天心率分布(最低/平均/最高)，反映日常心血管负荷。",
    },
    "body_mass": {
        "name": "体重",
        "unit": "kg",
        "desc": "体重变化趋势，需结合BMI和体脂率综合评估。",
    },
    "bmi": {
        "name": "BMI",
        "unit": "",
        "desc": "体质指数，评估体重与身高的比例关系。中国标准：18.5-23.9为正常。",
    },
    "body_fat": {
        "name": "体脂率",
        "unit": "%",
        "desc": "体脂肪占体重的百分比。成年男性正常范围10-20%。",
        "good": "lower",
    },
    "lean_mass": {
        "name": "瘦体重",
        "unit": "kg",
        "desc": "除脂肪外的体重，包括肌肉、骨骼、器官和水分。",
        "good": "higher",
    },
    "steps": {
        "name": "每日步数",
        "unit": "步",
        "desc": "每日总步数，世界卫生组织建议成人每日7500-10000步。",
        "good": "higher",
    },
    "active_energy": {
        "name": "活动能量",
        "unit": "kcal",
        "desc": "主动运动消耗的能量，不包括基础代谢。",
        "good": "higher",
    },
    "distance_wr": {
        "name": "步行/跑步距离",
        "unit": "km",
        "desc": "每日步行和跑步的总距离。",
    },
    "flights": {
        "name": "爬楼层数",
        "unit": "层",
        "desc": "每层约3米高度变化，反映日常垂直活动量。",
        "good": "higher",
    },
    "walking_speed": {
        "name": "步行速度",
        "unit": "km/h",
        "desc": "日常步行平均速度，是综合健康指标。健康成年男性约4.5-5.5 km/h。",
        "good": "higher",
    },
    "step_length": {
        "name": "步幅长度",
        "unit": "cm",
        "desc": "每步的距离，反映下肢力量和灵活性。正常范围65-85 cm。",
    },
    "double_support": {
        "name": "双支撑时间",
        "unit": "%",
        "desc": "步行时双脚同时着地的时间占比。数值越低说明平衡性越好，正常20-30%。",
        "good": "lower",
    },
    "asymmetry": {
        "name": "步态不对称",
        "unit": "%",
        "desc": "左右腿步行模式的差异程度。接近0%为理想，持续>10%需关注。",
        "good": "lower",
    },
    "steadiness": {
        "name": "步行稳定性",
        "unit": "",
        "desc": "Apple评估的步行稳定性等级，反映跌倒风险。",
    },
    "spo2": {
        "name": "血氧饱和度",
        "unit": "%",
        "desc": "血液中氧气的饱和程度。正常≥95%，低于90%为低氧血症。",
        "good": "higher",
    },
    "resp_rate": {
        "name": "呼吸频率",
        "unit": "次/分",
        "desc": "安静时每分钟呼吸次数。正常范围12-20次/分。",
    },
}


def _compute_trends(data: dict) -> dict:
    trends = {}
    for key, info in METRIC_INFO.items():
        records = data.get(key, [])
        if not records or not isinstance(records[0], dict) or "mean" not in records[0]:
            continue
        recent = [r for r in records if r.get("date", "") >= "2025-10-01"]
        prior = [r for r in records if "2025-04-01" <= r.get("date", "") < "2025-10-01"]

        recent_vals = [r["mean"] for r in recent if r.get("mean") is not None]
        prior_vals = [r["mean"] for r in prior if r.get("mean") is not None]

        if not recent_vals or not prior_vals:
            trends[key] = {"trend": "数据不足", "arrow": "→", "color": "#7f8c8d"}
            continue

        recent_avg = sum(recent_vals) / len(recent_vals)
        prior_avg = sum(prior_vals) / len(prior_vals)
        diff = recent_avg - prior_avg
        pct = (diff / prior_avg * 100) if prior_avg != 0 else 0

        good_dir = info.get("good", "")
        if abs(pct) < 2:
            arrow, color, word = "→", "#7f8c8d", "持平"
        elif diff > 0:
            if good_dir == "higher":
                arrow, color = "↑", "#27ae60"
            elif good_dir == "lower":
                arrow, color = "↑", "#e74c3c"
            else:
                arrow, color = "↑", "#f39c12"
            word = f"+{pct:.1f}%"
        else:
            if good_dir == "lower":
                arrow, color = "↓", "#27ae60"
            elif good_dir == "higher":
                arrow, color = "↓", "#e74c3c"
            else:
                arrow, color = "↓", "#f39c12"
            word = f"{pct:.1f}%"

        trends[key] = {
            "trend": f"近半年 vs 前半年: {word}",
            "arrow": arrow,
            "color": color,
            "recent_avg": round(recent_avg, 1),
            "prior_avg": round(prior_avg, 1),
        }
    return trends


OUTPUT_HTML = Path(__file__).parent / "dashboard.html"


def _process_daily(df, metric_type: str) -> list[dict]:
    daily = aggregate_daily(df, metric_type)
    daily = add_rolling_averages(daily)
    return daily.to_dict("records")


def _process_daily_sum(df, metric_type: str) -> list[dict]:
    sub = df[df["type"] == metric_type].copy()
    if sub.empty:
        return []
    daily = (
        sub.groupby("date")["value"]
        .agg(mean="sum", min="min", max="max", count="count")
        .reset_index()
    )
    daily["mean"] = daily["mean"].round(3)
    daily["min"] = daily["min"].round(3)
    daily["max"] = daily["max"].round(3)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily = add_rolling_averages(daily)
    return daily.to_dict("records")


def _safe_load(filename: str) -> pd.DataFrame:
    try:
        df = load_csv(filename)
        logging.info("Loaded %s: %d rows", filename, len(df))
        return df
    except Exception as e:
        logging.warning("Skipping %s: %s", filename, e)
        cols = pd.Index(["type", "value", "unit", "startdate", "sourcename", "date"])
        return pd.DataFrame(columns=cols)


def _date_range(values: Iterable[str]) -> str:
    items = [v for v in values if v]
    if not items:
        return "暂无数据"
    return f"{min(items)} ~ {max(items)}"


def _collect_dates(data: dict) -> list[str]:
    dates: list[str] = []
    for value in data.values():
        if isinstance(value, list) and value:
            if isinstance(value[0], dict) and "date" in value[0]:
                dates.extend([v.get("date") for v in value if v.get("date")])
    return dates


def build_data() -> dict:
    data: dict = {}

    df = _safe_load("1_Heart_Metrics.csv")
    df = _filter_recent(df)
    data["resting_hr"] = _remove_outliers(_process_daily(df, RESTING_HR))
    data["heart_rate"] = _remove_outliers(_process_daily(df, HEART_RATE))
    data["hrv_sdnn"] = _remove_outliers(_process_daily(df, HRV_SDNN))
    data["hr_recovery"] = _remove_outliers(_process_daily(df, HR_RECOVERY))
    data["walking_hr"] = _remove_outliers(_process_daily(df, WALKING_HR))
    del df

    df = _safe_load("2_Body_Composition.csv")
    df = _filter_recent(df)
    data["body_mass"] = _remove_outliers(_process_daily(df, BODY_MASS))
    data["bmi"] = _remove_outliers(_process_daily(df, BMI))
    data["body_fat"] = _remove_outliers(
        scale_records(_process_daily(df, BODY_FAT), 100)
    )
    data["lean_mass"] = _remove_outliers(_process_daily(df, LEAN_MASS))
    del df

    df1 = _safe_load("3_Activity_Energy_Part1.csv")
    df2 = _safe_load("3_Activity_Energy_Part2.csv")
    df3 = _safe_load("3_Activity_Energy_Part3.csv")
    df1 = _filter_recent(df1)
    df2 = _filter_recent(df2)
    df3 = _filter_recent(df3)
    df = pd.concat([df1, df2, df3], ignore_index=True)
    df = df.drop_duplicates(subset=["startdate", "type", "value", "sourcename"])
    df = deduplicate_activity_sources(df)
    data["steps"] = _remove_outliers(_process_daily_sum(df, STEPS))
    data["active_energy"] = _remove_outliers(_process_daily_sum(df, ACTIVE_ENERGY))
    data["basal_energy"] = _remove_outliers(_process_daily_sum(df, BASAL_ENERGY))
    data["distance_wr"] = _remove_outliers(_process_daily_sum(df, DISTANCE_WR))
    data["flights"] = _remove_outliers(_process_daily_sum(df, FLIGHTS))
    data["cycling"] = _remove_outliers(_process_daily_sum(df, CYCLING))
    data["six_min_walk"] = _remove_outliers(_process_daily(df, SIX_MIN_WALK))
    del df, df1, df2, df3

    df = _safe_load("4_Sleep_Analysis.csv")
    df = _filter_recent(df)
    data["sleep"] = reconstruct_sleep_sessions(df)
    data["sleep"] = [
        s
        for s in data["sleep"]
        if s.get("date", "") >= CUTOFF_DATE and s["total_hours"] <= 16
    ]
    del df

    df = _safe_load("5_Mobility_Gait.csv")
    df = _filter_recent(df)
    data["walking_speed"] = _remove_outliers(_process_daily(df, WALKING_SPEED))
    data["step_length"] = _remove_outliers(_process_daily(df, STEP_LENGTH))
    data["double_support"] = _remove_outliers(
        scale_records(_process_daily(df, DOUBLE_SUPPORT), 100)
    )
    data["asymmetry"] = _remove_outliers(
        scale_records(_process_daily(df, ASYMMETRY), 100)
    )
    data["steadiness"] = _remove_outliers(_process_daily(df, STEADINESS))
    del df

    df = _safe_load("7_Vitals_Respiratory.csv")
    df = _filter_recent(df)
    data["spo2"] = _remove_outliers(scale_records(_process_daily(df, SPO2), 100))
    data["resp_rate"] = _remove_outliers(_process_daily(df, RESP_RATE))
    del df

    df = _safe_load("8_Running_Dynamics.csv")
    df = _filter_recent(df)
    data["running_sessions"] = group_running_sessions(df)
    data["running_sessions"] = [
        s for s in data["running_sessions"] if s.get("date", "") >= CUTOFF_DATE
    ]
    del df

    df = _safe_load("10_Swimming_Water_Stats.csv")
    df = _filter_recent(df)
    data["swimming_sessions"] = group_swimming_sessions(df)
    data["swimming_sessions"] = [
        s for s in data["swimming_sessions"] if s.get("date", "") >= CUTOFF_DATE
    ]
    del df

    data["overview"] = compute_overview_stats(data)
    data["yearly_summary"] = compute_yearly_summary(
        data["resting_hr"],
        data["steps"],
        data["sleep"],
        data.get("bmi", []),
        data["running_sessions"],
        data["swimming_sessions"],
    )
    data["data_density"] = compute_data_density(
        {
            k: v
            for k, v in data.items()
            if isinstance(v, list) and v and isinstance(v[0], dict) and "date" in v[0]
        }
    )
    data["metric_info"] = METRIC_INFO
    data["trends"] = _compute_trends(data)

    return data


def build_html(data: dict) -> str:
    dates = _collect_dates(data)
    date_range = _date_range(dates)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_str = json.dumps(data, ensure_ascii=False, allow_nan=False)
    json_str = json_str.replace("</", "<\\/")

    html = HTML_TEMPLATE
    html = html.replace("__DATA__", json_str)
    html = html.replace("__DATE_RANGE__", date_range)
    html = html.replace("__GENERATED_AT__", generated_at)
    return html


def main() -> None:
    start = time.time()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    logging.info("Build started")
    data = build_data()
    logging.info("Data processing completed")
    html = build_html(data)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    logging.info("Wrote dashboard.html: %.2f MB", OUTPUT_HTML.stat().st_size / 1e6)
    logging.info("JSON size: %.2f MB", len(json.dumps(data, ensure_ascii=False)) / 1e6)
    logging.info("Build finished in %.1f sec", time.time() - start)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>Apple Health 健康数据分析</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --header: #1f2a3b;
      --text: #2c3e50;
      --muted: #7f8c8d;
      --border: #e6ebf2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      background: var(--header);
      color: #fff;
      padding: 20px 32px;
      font-size: 22px;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }
    header span {
      font-size: 14px;
      color: #cbd5e1;
    }
    nav {
      background: #fff;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .tabs {
      display: flex;
      gap: 8px;
      padding: 10px 24px;
      overflow-x: auto;
    }
    .tab-button {
      border: none;
      padding: 8px 16px;
      border-radius: 20px;
      background: #eef2f6;
      color: #34495e;
      cursor: pointer;
      font-size: 14px;
      white-space: nowrap;
    }
    .tab-button.active {
      background: #1f2a3b;
      color: #fff;
    }
    main {
      padding: 24px;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .panel-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }
    .chart {
      width: 100%;
      height: 400px;
      background: var(--card);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(30, 41, 59, 0.06);
      padding: 12px;
    }
    .chart-desc {
      font-size: 13px;
      color: var(--muted);
      padding: 8px 12px 0;
      line-height: 1.6;
    }
    .chart-trend {
      font-size: 14px;
      font-weight: 500;
      padding: 4px 12px 8px;
    }
    .trend-arrow {
      font-size: 16px;
    }
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }
    .metric-card {
      background: var(--card);
      padding: 18px;
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(30, 41, 59, 0.06);
    }
    .metric-card h4 {
      margin: 0 0 6px 0;
      font-size: 14px;
      color: var(--muted);
    }
    .metric-card .value {
      font-size: 26px;
      font-weight: 600;
    }
    .metric-card .sub {
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(30, 41, 59, 0.06);
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 13px;
    }
    th {
      background: #f1f5f9;
      font-weight: 600;
      cursor: pointer;
    }
    footer {
      padding: 16px 24px 32px;
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 768px) {
      header { font-size: 18px; }
      .chart { height: 320px; }
    }
  </style>
</head>
<body>
  <header>
    <div>Apple Health 健康数据分析</div>
    <span>数据范围：__DATE_RANGE__</span>
  </header>
  <nav>
    <div class="tabs">
      <button class="tab-button active" data-tab="overview">总览</button>
      <button class="tab-button" data-tab="heart">心脏</button>
      <button class="tab-button" data-tab="body">身体成分</button>
      <button class="tab-button" data-tab="activity">活动与能量</button>
      <button class="tab-button" data-tab="sleep">睡眠</button>
      <button class="tab-button" data-tab="mobility">步态</button>
      <button class="tab-button" data-tab="vitals">生命体征</button>
      <button class="tab-button" data-tab="running">跑步</button>
      <button class="tab-button" data-tab="swimming">游泳</button>
    </div>
  </nav>
  <main>
    <section class="tab-panel active" id="tab-overview">
      <div class="panel-grid">
        <div class="chart" id="chart-overview-heatmap"></div>
        <div class="cards-grid" id="overview-cards"></div>
        <table id="overview-table"></table>
      </div>
    </section>
    <section class="tab-panel" id="tab-heart">
      <div class="panel-grid">
        <div class="chart" id="chart-resting-hr"></div>
        <div class="chart" id="chart-hrv"></div>
        <div class="chart" id="chart-hr-recovery"></div>
        <div class="chart" id="chart-walking-hr"></div>
        <div class="chart" id="chart-hr-distribution"></div>
      </div>
    </section>
    <section class="tab-panel" id="tab-body">
      <div class="panel-grid">
        <div class="chart" id="chart-weight"></div>
        <div class="chart" id="chart-bmi"></div>
        <div class="chart" id="chart-bodyfat"></div>
        <div class="chart" id="chart-leanmass"></div>
      </div>
    </section>
    <section class="tab-panel" id="tab-activity">
      <div class="panel-grid">
        <div class="chart" id="chart-steps"></div>
        <div class="chart" id="chart-active-energy"></div>
        <div class="chart" id="chart-distance"></div>
        <div class="chart" id="chart-flights"></div>
        <div class="chart" id="chart-six-min-walk"></div>
      </div>
    </section>
    <section class="tab-panel" id="tab-sleep">
      <div class="panel-grid">
        <div class="chart" id="chart-sleep-duration"></div>
        <div class="chart" id="chart-sleep-stages"></div>
        <div class="chart" id="chart-sleep-efficiency"></div>
        <div class="chart" id="chart-sleep-pattern"></div>
        <div class="chart" id="chart-sleep-weekday"></div>
      </div>
    </section>
    <section class="tab-panel" id="tab-mobility">
      <div class="panel-grid">
        <div class="chart" id="chart-walking-speed"></div>
        <div class="chart" id="chart-step-length"></div>
        <div class="chart" id="chart-double-support"></div>
        <div class="chart" id="chart-asymmetry"></div>
        <div class="chart" id="chart-steadiness"></div>
      </div>
    </section>
    <section class="tab-panel" id="tab-vitals">
      <div class="panel-grid">
        <div class="chart" id="chart-spo2"></div>
        <div class="chart" id="chart-resp"></div>
      </div>
    </section>
    <section class="tab-panel" id="tab-running">
      <div class="panel-grid">
        <div class="chart" id="chart-run-pace"></div>
        <div class="chart" id="chart-run-stride"></div>
        <div class="chart" id="chart-run-power"></div>
        <div class="chart" id="chart-run-gct"></div>
        <div class="chart" id="chart-run-vosc"></div>
        <table id="running-table"></table>
      </div>
    </section>
    <section class="tab-panel" id="tab-swimming">
      <div class="panel-grid">
        <div class="chart" id="chart-swim-distance"></div>
        <div class="chart" id="chart-swim-strokes"></div>
        <div class="chart" id="chart-swim-frequency"></div>
        <div class="chart" id="chart-swim-pace"></div>
      </div>
    </section>
  </main>
  <footer>生成时间：__GENERATED_AT__</footer>
  <script type="application/json" id="health-data">__DATA__</script>
  <script>
    const healthData = JSON.parse(document.getElementById('health-data').textContent);
    const COLORS = {
      heart: '#E74C3C',
      body: '#F39C12',
      activity: '#27AE60',
      sleep: '#8E44AD',
      mobility: '#2980B9',
      vitals: '#1ABC9C',
      running: '#E67E22',
      swimming: '#3498DB'
    };

    const formatValue = (val, unit = '') => (val === null || val === undefined) ? '—' : `${val}${unit}`;
    const toDates = records => records.map(r => r.date);
    const toValues = (records, key) => records.map(r => r[key]);
    const toSeries = (records, key) => records.map(r => [r.date, r[key]]);

    function createChart(domId) {
      const el = document.getElementById(domId);
      return echarts.init(el, null, { renderer: 'canvas' });
    }

    function withDataZoom(option) {
      option.dataZoom = [
        { type: 'slider', xAxisIndex: 0, height: 18, bottom: 8 },
        { type: 'inside', xAxisIndex: 0 }
      ];
      option.toolbox = { feature: { saveAsImage: {} } };
      return option;
    }

    function lineOption({ title, color, dates, series, yName, markArea, markLine, yAxis } ) {
      if (markArea && series.length > 0) series[0].markArea = markArea;
      if (markLine && series.length > 0) series[0].markLine = markLine;
      return withDataZoom({
        title: { text: title, left: 'center' },
        color: [color, '#95a5a6', '#34495e'],
        tooltip: { trigger: 'axis' },
        legend: { top: 28 },
        xAxis: { type: 'category', data: dates },
        yAxis: yAxis || { type: 'value', name: yName },
        series
      });
    }

    function markAreaBand(y1, y2, color) {
      return {
        data: [[{ yAxis: y1 }, { yAxis: y2 }]],
        itemStyle: { color, opacity: 0.15 }
      };
    }

    function chartSubtitle(key) {
      const info = healthData.metric_info?.[key];
      const trend = healthData.trends?.[key];
      if (!info) return '';
      let html = `<div class="chart-desc">${info.desc}</div>`;
      if (trend && trend.trend !== '数据不足') {
        html += `<div class="chart-trend" style="color:${trend.color}">
          <span class="trend-arrow">${trend.arrow}</span> ${trend.trend}
          （近半年均值 ${trend.recent_avg}${info.unit}，上半年 ${trend.prior_avg}${info.unit}）
        </div>`;
      }
      return html;
    }

    function initOverview() {
      const density = healthData.data_density || [];
      const densityData = density.map(d => [d.date, d.count]);
      const range = density.length ? [density[0].date, density[density.length - 1].date] : [];
      const heatmap = createChart('chart-overview-heatmap');
      heatmap.setOption({
        title: { text: '数据密度热力图', left: 'center' },
        tooltip: { position: 'top' },
        visualMap: {
          min: 0,
          max: 10,
          orient: 'horizontal',
          left: 'center',
          bottom: 10
        },
        calendar: {
          range,
          cellSize: ['auto', 16],
          top: 60,
          left: 30,
          right: 30
        },
        series: [{
          type: 'heatmap',
          coordinateSystem: 'calendar',
          data: densityData
        }]
      });

      const overview = healthData.overview || {};
      const cards = [
        { label: '最新静息心率', value: formatValue(overview.latest_resting_hr, ' bpm'), sub: `趋势 ${overview.trend_resting_hr || '—'}` },
        { label: '最新 BMI', value: formatValue(overview.latest_bmi, '') },
        { label: '近30天平均步数', value: formatValue(overview.avg_steps_30d, ' 步') },
        { label: '近30天平均睡眠', value: formatValue(overview.avg_sleep_hours_30d, ' 小时') },
        { label: '近30天平均血氧', value: formatValue(overview.avg_spo2_30d, ' %') },
        { label: '最新步行速度', value: formatValue(overview.latest_walking_speed, ' km/h') },
        { label: '近90天跑步次数', value: formatValue(overview.running_sessions_90d, ' 次') },
        { label: '累计游泳次数', value: formatValue(overview.total_swimming_sessions, ' 次') }
      ];
      const cardWrap = document.getElementById('overview-cards');
      cardWrap.innerHTML = cards.map(c => `
        <div class="metric-card">
          <h4>${c.label}</h4>
          <div class="value">${c.value}</div>
          <div class="sub">${c.sub || ''}</div>
        </div>
      `).join('');

      const table = document.getElementById('overview-table');
      const rows = healthData.yearly_summary || [];
      table.innerHTML = `
        <thead>
          <tr>
            <th>年份</th>
            <th>平均静息心率</th>
            <th>平均步数</th>
            <th>平均睡眠小时</th>
            <th>平均 BMI</th>
            <th>跑步次数</th>
            <th>游泳次数</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => `
            <tr>
              <td>${r.year}</td>
              <td>${formatValue(r.avg_resting_hr)}</td>
              <td>${formatValue(r.avg_steps)}</td>
              <td>${formatValue(r.avg_sleep_hours)}</td>
              <td>${formatValue(r.avg_bmi)}</td>
              <td>${formatValue(r.running_sessions)}</td>
              <td>${formatValue(r.swimming_sessions)}</td>
            </tr>
          `).join('')}
        </tbody>
      `;
    }

    function initHeart() {
      const resting = healthData.resting_hr || [];
      const restingChart = createChart('chart-resting-hr');
      restingChart.setOption(lineOption({
        title: '静息心率趋势',
        color: COLORS.heart,
        dates: toDates(resting),
        series: [
          { name: '日均', type: 'line', data: toValues(resting, 'mean'), smooth: true },
          { name: 'MA7', type: 'line', data: toValues(resting, 'ma7'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(resting, 'ma30'), smooth: true }
        ],
        yName: 'bpm',
        markArea: markAreaBand(60, 100, COLORS.heart)
      }));
      document.getElementById('chart-resting-hr').insertAdjacentHTML('beforebegin', chartSubtitle('resting_hr'));

      const hrv = healthData.hrv_sdnn || [];
      const hrvChart = createChart('chart-hrv');
      hrvChart.setOption(lineOption({
        title: 'HRV SDNN (ms)',
        color: COLORS.heart,
        dates: toDates(hrv),
        series: [
          { name: '日均', type: 'line', data: toValues(hrv, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(hrv, 'ma30'), smooth: true }
        ],
        yName: 'ms',
        markArea: markAreaBand(40, 80, '#f8c9c5')
      }));
      document.getElementById('chart-hrv').insertAdjacentHTML('beforebegin', chartSubtitle('hrv_sdnn'));

      const recovery = healthData.hr_recovery || [];
      const recoveryChart = createChart('chart-hr-recovery');
      recoveryChart.setOption(withDataZoom({
        title: { text: '心率恢复 (1分钟)', left: 'center' },
        color: [COLORS.heart],
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: toDates(recovery) },
        yAxis: { type: 'value', name: 'bpm' },
        series: [{ name: '恢复值', type: 'scatter', data: toValues(recovery, 'mean'), markLine: { data: [{ yAxis: 12 }], label: { formatter: '12 bpm' } } }]
      }));
      document.getElementById('chart-hr-recovery').insertAdjacentHTML('beforebegin', chartSubtitle('hr_recovery'));

      const walkingHr = healthData.walking_hr || [];
      const walkingChart = createChart('chart-walking-hr');
      walkingChart.setOption(lineOption({
        title: '步行心率平均',
        color: COLORS.heart,
        dates: toDates(walkingHr),
        series: [
          { name: '日均', type: 'line', data: toValues(walkingHr, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(walkingHr, 'ma30'), smooth: true }
        ],
        yName: 'bpm'
      }));
      document.getElementById('chart-walking-hr').insertAdjacentHTML('beforebegin', chartSubtitle('walking_hr'));

      const hr = healthData.heart_rate || [];
      const minVals = toValues(hr, 'min');
      const maxVals = toValues(hr, 'max');
      const bandVals = maxVals.map((v, i) => (v !== null && minVals[i] !== null) ? v - minVals[i] : null);
      const distChart = createChart('chart-hr-distribution');
      distChart.setOption(withDataZoom({
        title: { text: '每日心率分布', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { top: 28 },
        xAxis: { type: 'category', data: toDates(hr) },
        yAxis: { type: 'value', name: 'bpm' },
        series: [
          { name: '最小', type: 'line', data: minVals, stack: 'range', smooth: true },
          { name: '区间', type: 'line', data: bandVals, stack: 'range', areaStyle: { opacity: 0.2 }, smooth: true },
          { name: '平均', type: 'line', data: toValues(hr, 'mean'), smooth: true }
        ]
      }));
      document.getElementById('chart-hr-distribution').insertAdjacentHTML('beforebegin', chartSubtitle('heart_rate'));
    }

    function initBody() {
      const weight = healthData.body_mass || [];
      const weightChart = createChart('chart-weight');
      weightChart.setOption(lineOption({
        title: '体重趋势',
        color: COLORS.body,
        dates: toDates(weight),
        series: [
          { name: '体重', type: 'scatter', data: toValues(weight, 'mean') },
          { name: 'MA30', type: 'line', data: toValues(weight, 'ma30'), smooth: true }
        ],
        yName: 'kg'
      }));
      document.getElementById('chart-weight').insertAdjacentHTML('beforebegin', chartSubtitle('body_mass'));

      const bmi = healthData.bmi || [];
      const bmiChart = createChart('chart-bmi');
      bmiChart.setOption(lineOption({
        title: 'BMI 趋势',
        color: COLORS.body,
        dates: toDates(bmi),
        series: [
          { name: 'BMI', type: 'scatter', data: toValues(bmi, 'mean') },
          { name: 'MA30', type: 'line', data: toValues(bmi, 'ma30'), smooth: true }
        ],
        yName: 'BMI',
        markArea: markAreaBand(18.5, 23.9, COLORS.body),
        markLine: { data: [{ yAxis: 18.5 }, { yAxis: 23.9 }, { yAxis: 27.9 }, { yAxis: 28 }] }
      }));
      document.getElementById('chart-bmi').insertAdjacentHTML('beforebegin', chartSubtitle('bmi'));

      const bodyFat = healthData.body_fat || [];
      const fatChart = createChart('chart-bodyfat');
      fatChart.setOption(lineOption({
        title: '体脂率趋势',
        color: COLORS.body,
        dates: toDates(bodyFat),
        series: [
          { name: '体脂率', type: 'scatter', data: toValues(bodyFat, 'mean') },
          { name: 'MA30', type: 'line', data: toValues(bodyFat, 'ma30'), smooth: true }
        ],
        yName: '%',
        markArea: markAreaBand(10, 20, '#fde8b4')
      }));
      document.getElementById('chart-bodyfat').insertAdjacentHTML('beforebegin', chartSubtitle('body_fat'));

      const lean = healthData.lean_mass || [];
      const leanChart = createChart('chart-leanmass');
      leanChart.setOption(lineOption({
        title: '瘦体重趋势',
        color: COLORS.body,
        dates: toDates(lean),
        series: [
          { name: '瘦体重', type: 'line', data: toValues(lean, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(lean, 'ma30'), smooth: true }
        ],
        yName: 'kg'
      }));
      document.getElementById('chart-leanmass').insertAdjacentHTML('beforebegin', chartSubtitle('lean_mass'));
    }

    function initActivity() {
      const steps = healthData.steps || [];
      const stepsChart = createChart('chart-steps');
      stepsChart.setOption(withDataZoom({
        title: { text: '每日步数', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { top: 28 },
        xAxis: { type: 'category', data: toDates(steps) },
        yAxis: { type: 'value', name: '步' },
        series: [
          {
            name: '步数',
            type: 'bar',
            data: toValues(steps, 'mean'),
            markLine: { data: [{ yAxis: 7500 }, { yAxis: 10000 }] },
            itemStyle: {
              color: params => {
                const v = params.value || 0;
                if (v < 5000) return '#e74c3c';
                if (v < 7500) return '#f1c40f';
                if (v < 10000) return '#a3d977';
                return '#27ae60';
              }
            }
          },
          { name: 'MA7', type: 'line', data: toValues(steps, 'ma7'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(steps, 'ma30'), smooth: true }
        ]
      }));
      document.getElementById('chart-steps').insertAdjacentHTML('beforebegin', chartSubtitle('steps'));

      const active = healthData.active_energy || [];
      const activeChart = createChart('chart-active-energy');
      activeChart.setOption(lineOption({
        title: '活动能量消耗',
        color: COLORS.activity,
        dates: toDates(active),
        series: [
          { name: 'kcal', type: 'line', data: toValues(active, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(active, 'ma30'), smooth: true }
        ],
        yName: 'kcal'
      }));
      document.getElementById('chart-active-energy').insertAdjacentHTML('beforebegin', chartSubtitle('active_energy'));

      const distance = healthData.distance_wr || [];
      const distChart = createChart('chart-distance');
      distChart.setOption(lineOption({
        title: '步行/跑步距离',
        color: COLORS.activity,
        dates: toDates(distance),
        series: [
          { name: '距离', type: 'line', data: toValues(distance, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(distance, 'ma30'), smooth: true }
        ],
        yName: 'km'
      }));
      document.getElementById('chart-distance').insertAdjacentHTML('beforebegin', chartSubtitle('distance_wr'));

      const flights = healthData.flights || [];
      const flightChart = createChart('chart-flights');
      flightChart.setOption(withDataZoom({
        title: { text: '爬楼层数', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: toDates(flights) },
        yAxis: { type: 'value', name: '层' },
        series: [{ name: '楼层', type: 'bar', data: toValues(flights, 'mean'), itemStyle: { color: COLORS.activity } }]
      }));
      document.getElementById('chart-flights').insertAdjacentHTML('beforebegin', chartSubtitle('flights'));

      const six = healthData.six_min_walk || [];
      const sixChart = createChart('chart-six-min-walk');
      sixChart.setOption(withDataZoom({
        title: { text: '6分钟步行测试', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: toDates(six) },
        yAxis: { type: 'value', name: 'm' },
        series: [{ name: '距离', type: 'scatter', data: toValues(six, 'mean'), markArea: markAreaBand(400, 700, '#dff5df'), itemStyle: { color: COLORS.activity } }]
      }));
    }

    function initSleep() {
      const sleep = healthData.sleep || [];
      const sleepChart = createChart('chart-sleep-duration');
      sleepChart.setOption(lineOption({
        title: '睡眠时长',
        color: COLORS.sleep,
        dates: sleep.map(r => r.date),
        series: [
          { name: '睡眠小时', type: 'line', data: sleep.map(r => r.sleep_hours), smooth: true }
        ],
        yName: '小时',
        markArea: markAreaBand(7, 9, COLORS.sleep)
      }));

      const stageData = sleep.filter(r => r.has_stages);
      const stageChart = createChart('chart-sleep-stages');
      stageChart.setOption(withDataZoom({
        title: { text: '睡眠分期占比', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { top: 28 },
        xAxis: { type: 'category', data: stageData.map(r => r.date) },
        yAxis: { type: 'value', name: '%' },
        series: [
          { name: '深睡', type: 'line', stack: 'stage', areaStyle: {}, data: stageData.map(r => r.deep_pct) },
          { name: 'REM', type: 'line', stack: 'stage', areaStyle: {}, data: stageData.map(r => r.rem_pct) },
          { name: '核心', type: 'line', stack: 'stage', areaStyle: {}, data: stageData.map(r => r.core_pct) },
          { name: '清醒', type: 'line', stack: 'stage', areaStyle: {}, data: stageData.map(r => r.awake_pct) }
        ]
      }));

      const effChart = createChart('chart-sleep-efficiency');
      effChart.setOption(lineOption({
        title: '睡眠效率',
        color: COLORS.sleep,
        dates: sleep.map(r => r.date),
        series: [
          { name: '效率', type: 'line', data: sleep.map(r => r.efficiency), smooth: true }
        ],
        yName: '%',
        markLine: { data: [{ yAxis: 85 }] }
      }));

      const toHour = t => {
        if (!t) return null;
        const [h, m] = t.split(':').map(Number);
        return h + (m || 0) / 60;
      };
      const bed = sleep.map(r => toHour(r.bedtime));
      const wake = sleep.map(r => toHour(r.wake_time));
      const patternChart = createChart('chart-sleep-pattern');
      patternChart.setOption(withDataZoom({
        title: { text: '入睡与醒来时间', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { top: 28 },
        xAxis: { type: 'category', data: sleep.map(r => r.date) },
        yAxis: [
          { type: 'value', name: '入睡(小时)', min: 0, max: 24, inverse: true },
          { type: 'value', name: '醒来(小时)', min: 0, max: 24 }
        ],
        series: [
          { name: '入睡时间', type: 'line', data: bed, yAxisIndex: 0, smooth: true },
          { name: '醒来时间', type: 'line', data: wake, yAxisIndex: 1, smooth: true }
        ]
      }));

      const weekdayData = sleep.filter(r => r.weekday !== undefined);
      const weekday = weekdayData.filter(r => r.weekday < 5);
      const weekend = weekdayData.filter(r => r.weekday >= 5);
      const avg = (list, key) => {
        const vals = list.map(r => r[key]).filter(v => v !== null && v !== undefined);
        return vals.length ? +(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2) : null;
      };
      const sleepWeekChart = createChart('chart-sleep-weekday');
      sleepWeekChart.setOption({
        title: { text: '工作日 vs 周末', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { top: 28 },
        xAxis: { type: 'category', data: ['平均时长', '深睡%', 'REM%'] },
        yAxis: { type: 'value' },
        series: [
          { name: '工作日', type: 'bar', data: [avg(weekday, 'sleep_hours'), avg(weekday, 'deep_pct'), avg(weekday, 'rem_pct')] },
          { name: '周末', type: 'bar', data: [avg(weekend, 'sleep_hours'), avg(weekend, 'deep_pct'), avg(weekend, 'rem_pct')] }
        ]
      });
    }

    function initMobility() {
      const walking = healthData.walking_speed || [];
      const walkingChart = createChart('chart-walking-speed');
      walkingChart.setOption(lineOption({
        title: '步行速度',
        color: COLORS.mobility,
        dates: toDates(walking),
        series: [
          { name: '速度', type: 'line', data: toValues(walking, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(walking, 'ma30'), smooth: true }
        ],
        yName: 'km/h',
        markArea: markAreaBand(4.5, 5.5, '#d9ecff')
      }));
      document.getElementById('chart-walking-speed').insertAdjacentHTML('beforebegin', chartSubtitle('walking_speed'));

      const step = healthData.step_length || [];
      const stepChart = createChart('chart-step-length');
      stepChart.setOption(lineOption({
        title: '步幅长度',
        color: COLORS.mobility,
        dates: toDates(step),
        series: [
          { name: '步幅', type: 'line', data: toValues(step, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(step, 'ma30'), smooth: true }
        ],
        yName: 'cm',
        markArea: markAreaBand(65, 85, '#d9ecff')
      }));
      document.getElementById('chart-step-length').insertAdjacentHTML('beforebegin', chartSubtitle('step_length'));

      const ds = healthData.double_support || [];
      const dsChart = createChart('chart-double-support');
      dsChart.setOption(lineOption({
        title: '双支撑时间占比',
        color: COLORS.mobility,
        dates: toDates(ds),
        series: [
          { name: '占比', type: 'line', data: toValues(ds, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(ds, 'ma30'), smooth: true }
        ],
        yName: '%',
        markArea: markAreaBand(20, 30, '#d9ecff')
      }));
      document.getElementById('chart-double-support').insertAdjacentHTML('beforebegin', chartSubtitle('double_support'));

      const asym = healthData.asymmetry || [];
      const asymChart = createChart('chart-asymmetry');
      asymChart.setOption(lineOption({
        title: '步态不对称',
        color: COLORS.mobility,
        dates: toDates(asym),
        series: [
          { name: '不对称', type: 'line', data: toValues(asym, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(asym, 'ma30'), smooth: true }
        ],
        yName: '%',
        markLine: { data: [{ yAxis: 10 }] }
      }));
      document.getElementById('chart-asymmetry').insertAdjacentHTML('beforebegin', chartSubtitle('asymmetry'));

      const steadiness = healthData.steadiness || [];
      const steadinessChart = createChart('chart-steadiness');
      steadinessChart.setOption(lineOption({
        title: '步行稳定性',
        color: COLORS.mobility,
        dates: toDates(steadiness),
        series: [
          { name: '稳定性', type: 'line', data: toValues(steadiness, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(steadiness, 'ma30'), smooth: true }
        ],
        yName: '级别'
      }));
      document.getElementById('chart-steadiness').insertAdjacentHTML('beforebegin', chartSubtitle('steadiness'));
    }

    function initVitals() {
      const spo2 = healthData.spo2 || [];
      const spo2Chart = createChart('chart-spo2');
      spo2Chart.setOption(lineOption({
        title: '血氧饱和度',
        color: COLORS.vitals,
        dates: toDates(spo2),
        series: [
          { name: '日均', type: 'line', data: toValues(spo2, 'mean'), smooth: true },
          { name: 'MA7', type: 'line', data: toValues(spo2, 'ma7'), smooth: true }
        ],
        yName: '%',
        markArea: markAreaBand(95, 100, '#dff7f2'),
        markLine: { data: [{ yAxis: 95 }, { yAxis: 90 }] }
      }));
      document.getElementById('chart-spo2').insertAdjacentHTML('beforebegin', chartSubtitle('spo2'));

      const resp = healthData.resp_rate || [];
      const respChart = createChart('chart-resp');
      respChart.setOption(lineOption({
        title: '呼吸频率',
        color: COLORS.vitals,
        dates: toDates(resp),
        series: [
          { name: '日均', type: 'line', data: toValues(resp, 'mean'), smooth: true },
          { name: 'MA30', type: 'line', data: toValues(resp, 'ma30'), smooth: true }
        ],
        yName: '次/分',
        markArea: markAreaBand(12, 20, '#dff7f2')
      }));
      document.getElementById('chart-resp').insertAdjacentHTML('beforebegin', chartSubtitle('resp_rate'));
    }

    function initRunning() {
      const run = healthData.running_sessions || [];
      const dates = run.map(r => `${r.date} ${r.time}`);

      const paceChart = createChart('chart-run-pace');
      paceChart.setOption(withDataZoom({
        title: { text: '跑步配速', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: dates },
        yAxis: { type: 'value', name: 'min/km', inverse: true },
        series: [{ name: '配速', type: 'line', data: run.map(r => r.pace_min_per_km), smooth: true, itemStyle: { color: COLORS.running } }]
      }));

      const strideChart = createChart('chart-run-stride');
      strideChart.setOption(lineOption({
        title: '步幅长度',
        color: COLORS.running,
        dates,
        series: [{ name: '步幅', type: 'line', data: run.map(r => r.avg_stride_m), smooth: true }],
        yName: 'm'
      }));

      const powerChart = createChart('chart-run-power');
      powerChart.setOption(lineOption({
        title: '跑步功率',
        color: COLORS.running,
        dates,
        series: [{ name: '功率', type: 'line', data: run.map(r => r.avg_power_w), smooth: true }],
        yName: 'W'
      }));

      const gctChart = createChart('chart-run-gct');
      gctChart.setOption(lineOption({
        title: '触地时间',
        color: COLORS.running,
        dates,
        series: [{ name: 'GCT', type: 'line', data: run.map(r => r.avg_gct_ms), smooth: true }],
        yName: 'ms',
        markArea: markAreaBand(200, 350, '#fde5d0')
      }));

      const voscChart = createChart('chart-run-vosc');
      voscChart.setOption(lineOption({
        title: '垂直振幅',
        color: COLORS.running,
        dates,
        series: [{ name: '振幅', type: 'line', data: run.map(r => r.avg_vosc_cm), smooth: true }],
        yName: 'cm',
        markArea: markAreaBand(6, 12, '#fde5d0')
      }));

      const table = document.getElementById('running-table');
      const headers = [
        { key: 'date', label: '日期' },
        { key: 'duration_min', label: '时长(min)' },
        { key: 'avg_speed_mps', label: '平均速度(m/s)' },
        { key: 'pace_min_per_km', label: '配速(min/km)' },
        { key: 'avg_power_w', label: '功率(W)' }
      ];
      let sortKey = 'date';
      let sortAsc = true;
      const renderTable = () => {
        const rows = [...run].sort((a, b) => {
          if (a[sortKey] === b[sortKey]) return 0;
          return (a[sortKey] > b[sortKey] ? 1 : -1) * (sortAsc ? 1 : -1);
        });
        table.innerHTML = `
          <thead>
            <tr>
              ${headers.map(h => `<th data-key="${h.key}">${h.label}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${rows.map(r => `
              <tr>
                <td>${r.date} ${r.time}</td>
                <td>${formatValue(r.duration_min)}</td>
                <td>${formatValue(r.avg_speed_mps)}</td>
                <td>${formatValue(r.pace_min_per_km)}</td>
                <td>${formatValue(r.avg_power_w)}</td>
              </tr>
            `).join('')}
          </tbody>
        `;
        table.querySelectorAll('th').forEach(th => {
          th.addEventListener('click', () => {
            const key = th.dataset.key;
            if (key === sortKey) {
              sortAsc = !sortAsc;
            } else {
              sortKey = key;
              sortAsc = true;
            }
            renderTable();
          }, { once: true });
        });
      };
      renderTable();
    }

    function initSwimming() {
      const swim = healthData.swimming_sessions || [];
      const dates = swim.map(r => `${r.date} ${r.time}`);

      const distChart = createChart('chart-swim-distance');
      distChart.setOption(withDataZoom({
        title: { text: '游泳距离', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: dates },
        yAxis: { type: 'value', name: 'm' },
        series: [{ name: '距离', type: 'bar', data: swim.map(r => r.total_distance_m), itemStyle: { color: COLORS.swimming } }]
      }));

      const strokeChart = createChart('chart-swim-strokes');
      strokeChart.setOption(withDataZoom({
        title: { text: '划水次数', left: 'center' },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: dates },
        yAxis: { type: 'value', name: '次' },
        series: [{ name: '划水', type: 'bar', data: swim.map(r => r.total_strokes), itemStyle: { color: '#6bb6ff' } }]
      }));

      const swimDays = {};
      swim.forEach(r => {
        swimDays[r.date] = (swimDays[r.date] || 0) + 1;
      });
      const frequency = Object.entries(swimDays).map(([date, count]) => [date, count]);
      const range = frequency.length ? [frequency[0][0], frequency[frequency.length - 1][0]] : [];
      const freqChart = createChart('chart-swim-frequency');
      freqChart.setOption({
        title: { text: '游泳频率', left: 'center' },
        tooltip: { position: 'top' },
        visualMap: { min: 0, max: 3, orient: 'horizontal', left: 'center', bottom: 10 },
        calendar: { range, cellSize: ['auto', 16], top: 60, left: 30, right: 30 },
        series: [{ type: 'heatmap', coordinateSystem: 'calendar', data: frequency }]
      });

      const paceChart = createChart('chart-swim-pace');
      paceChart.setOption(lineOption({
        title: '游泳配速',
        color: COLORS.swimming,
        dates,
        series: [{ name: 'min/100m', type: 'line', data: swim.map(r => r.pace_min_per_100m), smooth: true }],
        yName: 'min/100m'
      }));
    }

    const initMap = {
      overview: initOverview,
      heart: initHeart,
      body: initBody,
      activity: initActivity,
      sleep: initSleep,
      mobility: initMobility,
      vitals: initVitals,
      running: initRunning,
      swimming: initSwimming
    };
    const inited = {};

    function activateTab(tab) {
      document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
      });
      document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `tab-${tab}`);
      });
      if (!inited[tab]) {
        initMap[tab]();
        inited[tab] = true;
      }
    }

    document.querySelectorAll('.tab-button').forEach(btn => {
      btn.addEventListener('click', () => activateTab(btn.dataset.tab));
    });
    activateTab('overview');

    window.addEventListener('resize', () => {
      document.querySelectorAll('.chart').forEach(el => {
        const chart = echarts.getInstanceByDom(el);
        if (chart) chart.resize();
      });
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
