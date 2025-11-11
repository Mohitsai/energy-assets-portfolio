# src/jobs/backfill_prices.py

import json
from typing import Optional, List, Dict

# import boto3
from src.io.store import make_store
import pandas as pd

from src.core.registry import Registry
from src.io.s3_writer import put_csv_df, put_parquet_df, uri_to_bucket_key

# s3 = boto3.client("s3")

# ---------- helpers: columns ----------

DATE_ALIASES = ["date", "datetime", "timestamp", "trade_date", "pricedate"]

LOWER_MAP = {
    "adj_close": ["adj_close", "adjclose", "adjusted_close", "adjusted", "adj_price", "adj_prc"],
    "close": ["close", "closing_price", "last", "settle", "settlement"],
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "volume": ["volume", "vol"],
}


def _flatten_columns(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Flatten MultiIndex columns and optionally drop the ticker level if it matches symbol."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = [str(p) for p in tup if p is not None and str(p) != ""]
            if symbol and len(parts) >= 2 and parts[1].lower() == str(symbol).lower():
                name = parts[0]
            else:
                name = "_".join(parts)
            new_cols.append(name)
        df.columns = new_cols
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    # remove quotes or stray commas that can appear after tuple stringification
    for ch in ["'", '"', ","]:
        s = s.replace(ch, "")
    for ch in [" ", "-", ".", "/", "\\", "(", ")", "*"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _normalize_cols(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    df = _flatten_columns(df, symbol=symbol)
    df.columns = [_norm(c) for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _map_price_col(df: pd.DataFrame, target: str) -> Optional[str]:
    return _find_col(df, LOWER_MAP.get(target, []))


# ---------- normalization to silver ----------

def normalize_to_silver(raw: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Normalize any adapter output to a strict silver schema.
    Output columns: date, adj_close, plus optional close, open, high, low, volume.
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("normalize_to_silver received empty DataFrame")

    df = _normalize_cols(raw, symbol=symbol)

    # If date not present as a column, try to lift it from the index
    date_col = _find_col(df, DATE_ALIASES)
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
            date_col = "date"
        elif df.index.name and df.index.name.lower() in DATE_ALIASES:
            df = df.reset_index()
            date_col = df.columns[0]

    if date_col is None:
        raise ValueError(f"Could not find a date column in {list(df.columns)}")

    adj_col = _map_price_col(df, "adj_close")
    close_col = _map_price_col(df, "close")
    open_col = _map_price_col(df, "open")
    high_col = _map_price_col(df, "high")
    low_col = _map_price_col(df, "low")
    vol_col = _map_price_col(df, "volume")

    if adj_col is None and close_col is not None:
        adj_col = close_col
    if adj_col is None:
        raise ValueError(f"Could not find adj_close or close in {list(df.columns)}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=False)

    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    out["adj_close"] = to_num(df[adj_col])

    if close_col and close_col != adj_col:
        out["close"] = to_num(df[close_col])
    if open_col:
        out["open"] = to_num(df[open_col])
    if high_col:
        out["high"] = to_num(df[high_col])
    if low_col:
        out["low"] = to_num(df[low_col])
    if vol_col:
        out["volume"] = to_num(df[vol_col])

    # clean rows
    out = out.dropna(subset=["date", "adj_close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    out = out[out["adj_close"] > 0]

    # normalize to midnight naive for deterministic partitioning
    out["date"] = out["date"].dt.tz_localize(None).dt.normalize()

    cols = ["date", "adj_close"]
    for c in ["close", "open", "high", "low", "volume"]:
        if c in out.columns:
            cols.append(c)
    return out[cols]


# ---------- QA ----------

def qa_summary(df: pd.DataFrame) -> Dict:
    if df is None or df.empty:
        return {"rows": 0, "start": None, "end": None, "columns": [], "null_counts": {}, "monotonic_dates": True}
    return {
        "rows": int(len(df)),
        "start": df["date"].min().strftime("%Y-%m-%d"),
        "end": df["date"].max().strftime("%Y-%m-%d"),
        "columns": list(df.columns),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "monotonic_dates": bool(df["date"].is_monotonic_increasing),
    }


# ---------- paths ----------

def bronze_base_path(reg: Registry, provider: str, symbol: str) -> str:
    """
    Ensure provider segment appears exactly once in the bronze prefix.
    """
    base = reg.bronze_prefix
    seg = f"provider={provider}"
    if seg not in base.split("/"):
        base = f"{base}/{seg}"
    return f"{base}/asset={symbol}"


# ---------- main ----------

def main():
    reg = Registry()
    store = reg.make_store()
    start, end = reg.backfill_window
    # bucket, _ = uri_to_bucket_key(reg.bucket_uri)

    for asset in reg.assets:
        symbol = asset["code"]
        provider = asset["provider"]
        adapter = reg.adapter_for(provider)

        print(f"fetching {symbol} from {start} to {end} via {provider}")
        raw = adapter.fetch_eod(symbol, start, end)

        # Ensure DataFrame even if adapter returns another structure
        if raw is None:
            print(f"no data for {symbol}")
            continue
        if not isinstance(raw, pd.DataFrame):
            raw = pd.DataFrame(raw)
        if raw.empty:
            print(f"no data for {symbol}")
            continue

        # Write bronze as-is, but if DatetimeIndex then reset for a clean CSV
        bronze_df = raw.copy()
        if isinstance(bronze_df.index, pd.DatetimeIndex) or bronze_df.index.name:
            bronze_df = bronze_df.reset_index()

        bronze_key = f"{bronze_base_path(reg, provider, symbol)}/full.csv"
        # put_csv_df(bucket, bronze_key, bronze_df)
        store.put_csv_df(bronze_key, bronze_df)
        # print(f"wrote s3://{bucket}/{bronze_key} rows {len(bronze_df)}")

        # Normalize to silver
        silver_df = normalize_to_silver(raw, symbol=symbol)
        silver_base = f"{reg.silver_prefix}/daily_prices/asset={symbol}"
        silver_key = f"{silver_base}/series.parquet"
        # put_parquet_df(bucket, silver_key, silver_df)
        store.put_parquet_df(silver_key, silver_df)
        # print(f"wrote s3://{bucket}/{silver_key} rows {len(silver_df)}")

        # QA
        qa = qa_summary(silver_df)
        qa_key = f"{silver_base}/qa.json"
        # s3.put_object(Bucket=bucket, Key=qa_key, Body=json.dumps(qa, indent=2).encode())
        store.put_json(qa_key, qa)
        print(f"QA {symbol} -> {qa}")


if __name__ == "__main__":
    main()
