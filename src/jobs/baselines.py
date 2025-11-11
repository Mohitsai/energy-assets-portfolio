import io
import json
import numpy as np
import pandas as pd
import boto3
from src.core.registry import Registry
from src.io.s3_writer import uri_to_bucket_key

s3 = boto3.client("s3")

def read_gold(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def metrics(returns: pd.Series):
    daily = returns.dropna()
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    # max drawdown
    eq = (1 + daily).cumprod()
    rolling_max = eq.cummax()
    dd = (eq / rolling_max - 1).min()
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "max_drawdown": float(dd)}

def equal_weight(df, assets):
    rets = df[[f"ret_1_{a}" for a in assets]].mean(axis=1)
    return rets.rename("eqw_ret")

def momentum_20d(df, assets):
    # rank by past 20 day return, top 2 equally weighted
    past = {}
    for a in assets:
        past[a] = (1 + df[f"ret_1_{a}"]).rolling(20).apply(lambda x: np.prod(x) - 1, raw=True)
    rank_df = pd.DataFrame(past)
    top2 = rank_df.apply(lambda row: list(row.nlargest(2).index), axis=1)
    weights = pd.DataFrame(0, index=df.index, columns=assets, dtype=float)
    for idx, winners in top2.items():
        for w in winners:
            weights.loc[idx, w] = 0.5
    port = (weights.values * df[[f"ret_1_{a}" for a in assets]].values).sum(axis=1)
    return pd.Series(port, index=df.index, name="mom20_ret")

def main():
    reg = Registry()
    store = reg.make_store()
    # bucket, _ = uri_to_bucket_key(reg.bucket_uri)
    feats_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/features_v1.parquet"
    # df = read_gold(bucket, feats_key)
    df = store.read_parquet(f"{reg.gold_prefix.strip('/')}/features_v1.parquet")
    assets = [a["code"] for a in reg.assets]

    eqw = equal_weight(df, assets)
    mom = momentum_20d(df, assets)

    # split train test
    split_date = pd.to_datetime(df["date"]).quantile(0.75)
    mask_test = pd.to_datetime(df["date"]) > split_date
    res = {
        "eqw_test": metrics(eqw[mask_test]),
        "mom20_test": metrics(mom[mask_test])
    }
    report_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/baseline_report.json"
    # s3.put_object(Bucket=bucket, Key=report_key, Body=json.dumps(res, indent=2).encode())
    store.put_json(f"{reg.gold_prefix.strip('/')}/baseline_report.json", res)
    print("baseline report", res)

if __name__ == "__main__":
    main()
