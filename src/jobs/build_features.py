import io
import json
from datetime import datetime
import pandas as pd
import numpy as np
import boto3
from src.core.registry import Registry

from src.core.registry import Registry
from src.io.s3_writer import uri_to_bucket_key, put_parquet_df

s3 = boto3.client("s3")

def read_parquet(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def sma(x, w):
    return x.rolling(w).mean()

def vol(x, w):
    return x.pct_change().rolling(w).std()

def build_features(panel: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # daily returns for each asset
    for a in assets:
        df[f"ret_1_{a}"] = df[a].pct_change()

    # simple technical features per asset
    for a in assets:
        df[f"sma_5_{a}"] = sma(df[a], 5)
        df[f"sma_20_{a}"] = sma(df[a], 20)
        df[f"vol_10_{a}"] = vol(df[a], 10)

    # normalize prices for scale stability
    for a in assets:
        df[f"zprice_{a}"] = (df[a] - df[a].rolling(60).mean()) / (df[a].rolling(60).std() + 1e-9)

    # drop early NaN rows
    df = df.dropna().reset_index(drop=True)

    # target next day returns for evaluation and supervised baselines
    for a in assets:
        df[f"next_ret_{a}"] = df[f"ret_1_{a}"].shift(-1)

    return df.dropna().reset_index(drop=True)

def main():
    reg = Registry()
    store = reg.make_store()
    # bucket, _ = uri_to_bucket_key(reg.bucket_uri)
    silver_key = f"{reg.silver_prefix}/panel.parquet"
    # panel = read_parquet(bucket, silver_key)
    panel = store.read_parquet(f"{reg.silver_prefix}/daily_prices/panel.parquet")

    assets = [a["code"] for a in reg.assets]
    feats = build_features(panel, assets)

    gold_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/features_v1.parquet"
    # put_parquet_df(bucket, gold_key, feats)
    store.put_parquet_df(f"{reg.gold_prefix.strip('/')}/features_v1.parquet", feats)

    # schema preview
    preview_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/features_v1_preview.json"
    cols = list(feats.columns)
    # s3.put_object(Bucket=bucket, Key=preview_key, Body=json.dumps({"columns": cols[:20], "total_cols": len(cols)}, indent=2).encode())
    # print(f"wrote s3://{bucket}/{gold_key} rows {len(feats)}")
    store.put_json(f"{reg.gold_prefix.strip('/')}/features_v1_preview.json", {"columns": cols[:20], "total_cols": len(cols)})

if __name__ == "__main__":
    main()
