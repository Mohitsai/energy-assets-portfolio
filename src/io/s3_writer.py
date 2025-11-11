import io, boto3, pandas as pd
from datetime import date

s3 = boto3.client("s3")

def put_csv_df(bucket: str, key: str, df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def put_parquet_df(bucket: str, key: str, df: pd.DataFrame):
    out = io.BytesIO()
    df.to_parquet(out, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=out.getvalue())

def uri_to_bucket_key(uri: str):
    assert uri.startswith("s3://")
    no = uri.replace("s3://", "")
    parts = no.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key
