import json
from datetime import date
import pandas as pd

def qa_prices_panel(panel: pd.DataFrame, required_cols: list[str]) -> dict:
    report = {"checks": {}, "summary": {}}
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date

    # freshness
    latest = panel["date"].max() if not panel.empty else None
    report["checks"]["freshness"] = {"latest_date": str(latest)}

    # completeness
    missing_by_date = {}
    for col in required_cols:
        missing = panel[col].isna().sum()
        missing_by_date[col] = int(missing)
    report["checks"]["missing_counts"] = missing_by_date

    # monotonic dates
    sorted_dates = panel["date"].sort_values()
    is_mono = all(a <= b for a, b in zip(sorted_dates, sorted_dates[1:]))
    report["checks"]["monotonic_dates"] = bool(is_mono)

    # basic outlier guard on returns
    ret = panel[required_cols].pct_change().stack()
    z = (ret - ret.mean()) / (ret.std() + 1e-12)
    extreme = int((z.abs() > 10).sum())
    report["checks"]["extreme_return_points"] = extreme

    report["summary"]["ok"] = bool(is_mono and extreme == 0)
    return report
