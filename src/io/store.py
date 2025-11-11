import os, io, json
from pathlib import Path
from typing import Optional
import pandas as pd

# Optional S3, only imported when needed
try:
    import boto3
except Exception:
    boto3 = None

class LocalStore:
    def __init__(self, root: str):
        self.root = Path(root)

    def _full(self, rel: str) -> Path:
        p = self.root / rel.strip("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_csv_df(self, rel_key: str, df: pd.DataFrame):
        p = self._full(rel_key)
        df.to_csv(p, index=False)

    def put_parquet_df(self, rel_key: str, df: pd.DataFrame):
        p = self._full(rel_key)
        df.to_parquet(p, index=False)

    def put_json(self, rel_key: str, obj: dict):
        p = self._full(rel_key)
        p.write_text(json.dumps(obj, indent=2))

    def read_csv(self, rel_key: str) -> pd.DataFrame:
        p = self.root / rel_key.strip("/")
        return pd.read_csv(p)

    def read_parquet(self, rel_key: str) -> pd.DataFrame:
        p = self.root / rel_key.strip("/")
        return pd.read_parquet(p)

class S3Store:
    def __init__(self, bucket: str):
        assert boto3 is not None, "boto3 required for S3 mode"
        self.s3 = boto3.client("s3")
        assert bucket.startswith("s3://")
        self.bucket = bucket.replace("s3://", "").split("/")[0]

    def put_csv_df(self, key: str, df: pd.DataFrame):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def put_parquet_df(self, key: str, df: pd.DataFrame):
        b = io.BytesIO()
        df.to_parquet(b, index=False)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=b.getvalue())

    def put_json(self, key: str, obj: dict):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(obj, indent=2).encode())

    def read_csv(self, key: str) -> pd.DataFrame:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    def read_parquet(self, key: str) -> pd.DataFrame:
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

def make_store(cfg: dict):
    backend = cfg["storage"].get("backend", "local")
    if backend == "local":
        return LocalStore(cfg["storage"]["local_root"])
    elif backend == "s3":
        return S3Store(cfg["storage"]["bucket"])
    else:
        raise ValueError(f"Unknown backend {backend}")
