import io
import json
import numpy as np
import pandas as pd
import boto3
from stable_baselines3 import PPO
from pathlib import Path

from src.core.registry import Registry
from src.io.s3_writer import uri_to_bucket_key
from src.rl.env_portfolio import PortfolioEnv

s3 = boto3.client("s3")

def read_parquet(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def load_model_from_s3(bucket, key, local="/tmp/model.zip"):
    obj = s3.get_object(Bucket=bucket, Key=key)
    with open(local, "wb") as f:
        f.write(obj["Body"].read())
    return PPO.load(local)

def last_n(df, n):
    return df.tail(n).reset_index(drop=True)

def main():
    reg = Registry()
    store = reg.make_store()
    # bucket, _ = uri_to_bucket_key(reg.bucket_uri)
    feats_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/features_v1.parquet"
    model_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/models/ppo_portfolio_model.zip"

    # df = read_parquet(bucket, feats_key)
    # assets = [a["code"] for a in reg.assets]
    # # use last window + 1 rows to build an env ending on latest date
    # window = 5
    # df_use = last_n(df, 200)  # small slice is enough

    # env = PortfolioEnv(df_use, assets, window=window, tc_bps=1.0, risk_lambda=0.1)
    # obs, _ = env.reset()
    # model = load_model_from_s3(bucket, model_key)

    df = store.read_parquet(f"{reg.gold_prefix.strip('/')}/features_v1.parquet")
    assets = [a["code"] for a in reg.assets]
    # use last window + 1 rows to build an env ending on latest date
    window = 5
    df_use = last_n(df, 200)  # small slice is enough

    env = PortfolioEnv(df_use, assets, window=window, tc_bps=1.0, risk_lambda=0.1)
    obs, _ = env.reset()

    model_path = f"{reg.gold_prefix.strip('/')}/models/ppo_portfolio_model.zip"

    local_model = Path("local/tmp/model.zip")
    local_model.parent.mkdir(parents=True, exist_ok=True)
    # if using LocalStore, model_path is under local_root already
    src_path = Path(reg.cfg["storage"]["local_root"]) / model_path
    local_model.write_bytes(src_path.read_bytes())

    model = PPO.load(str(local_model))


    # roll until final step to get current weights
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if done:
            w = info["weights"].tolist()
            break

    allocation = {a: float(wi) for a, wi in zip(assets, w)}
    out = {
        "date": str(pd.to_datetime(df_use["date"].iloc[-1]).date()),
        "allocation": allocation
    }
    # out_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/signals/allocation_latest.json"
    # s3.put_object(Bucket=bucket, Key=out_key, Body=json.dumps(out, indent=2).encode())
    store.put_json(f"{reg.gold_prefix.strip('/')}/signals/allocation_latest.json", out)
    print("allocation", out)

if __name__ == "__main__":
    main()
