import io
import os
import json
import boto3
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

from src.core.registry import Registry
from src.io.s3_writer import uri_to_bucket_key
from src.rl.env_portfolio import PortfolioEnv

s3 = boto3.client("s3")

def read_gold(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def make_env(df, assets, window=5, tc_bps=1.0, risk_lambda=0.0):
    def _thunk():
        return PortfolioEnv(df, assets, window=window, tc_bps=tc_bps, risk_lambda=risk_lambda)
    return _thunk

def split_train_test(df):
    q = pd.to_datetime(df["date"]).quantile(0.75)
    train = df[pd.to_datetime(df["date"]) <= q].reset_index(drop=True)
    test = df[pd.to_datetime(df["date"]) > q].reset_index(drop=True)
    return train, test

def main():
    reg = Registry()
    store = reg.make_store()
    # bucket, _ = uri_to_bucket_key(reg.bucket_uri)
    feats_key = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/features_v1.parquet"
    # df = read_gold(bucket, feats_key)
    df = store.read_parquet(f"{reg.gold_prefix.strip('/')}/features_v1.parquet")
    assets = [a["code"] for a in reg.assets]

    train_df, test_df = split_train_test(df)

    env = DummyVecEnv([make_env(train_df, assets, window=5, tc_bps=1.0, risk_lambda=0.1)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None)

    timesteps = int(2e5)
    model.learn(total_timesteps=timesteps)

    # quick validation equity
    test_env = PortfolioEnv(test_df, assets, window=5, tc_bps=1.0, risk_lambda=0.1)
    obs, _ = test_env.reset()
    port_rets = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        port_rets.append(info["port_ret"])
        if done:
            break

    
    daily = np.array(port_rets, dtype=float)
    ann_ret = daily.mean() * 252
    ann_vol = daily.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)

    report = {"ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
    key_report = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/rl_report.json"
    # s3.put_object(Bucket=bucket, Key=key_report, Body=json.dumps(report, indent=2).encode())
    print("rl report", report)

    # save model artifact to S3
    # local_path = "/tmp/ppo_portfolio_model.zip"
    # model.save(local_path)
    # key_model = f"{reg.cfg['storage']['gold_prefix'].strip('/')}/models/ppo_portfolio_model.zip"
    # with open(local_path, "rb") as f:
    #     s3.put_object(Bucket=bucket, Key=key_model, Body=f.read())
    # print(f"saved model to s3://{bucket}/{key_model}")

    local_path = "local/artifacts/ppo_portfolio_model.zip"
    Path("local/artifacts").mkdir(parents=True, exist_ok=True)
    model.save(local_path)
    store.put_json(f"{reg.gold_prefix.strip('/')}/rl_report.json", report)
    Path(reg.cfg["storage"]["local_root"]).mkdir(parents=True, exist_ok=True)

    # keep a copy in data area for your inference job
    from shutil import copyfile
    copyfile(local_path, f"{reg.cfg['storage']['local_root'].rstrip('/')}/{reg.gold_prefix.strip('/')}/models/ppo_portfolio_model.zip")


if __name__ == "__main__":
    main()
