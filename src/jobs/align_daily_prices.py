import boto3, pandas as pd, io, json
from datetime import date
from src.core.registry import Registry
from src.io.s3_writer import put_parquet_df, uri_to_bucket_key
from src.core.qa import qa_prices_panel

s3 = boto3.client("s3")

def read_bronze_csv(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def main():
    reg = Registry()
    bucket, _ = uri_to_bucket_key(reg.bucket_uri)

    frames = []
    names = []
    for a in reg.assets:
        symbol = a["code"]
        key = f"{reg.bronze_prefix}/asset={symbol}/full.csv"
        df = read_bronze_csv(bucket, key)
        df = df[["date","adj_close"]].rename(columns={"adj_close": symbol})
        frames.append(df)
        names.append(symbol)

    # inner join on dates to keep only common trading days
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on="date", how="inner")

    panel = panel.sort_values("date").reset_index(drop=True)

    # write silver Parquet
    silver_key = f"{reg.silver_prefix}/panel.parquet"
    put_parquet_df(bucket, silver_key, panel)
    print(f"wrote s3://{bucket}/{silver_key} rows {len(panel)}")

    # QA report
    report = qa_prices_panel(panel, names)
    qa_key = f"{reg.silver_prefix}/qa_report.json"
    s3.put_object(Bucket=bucket, Key=qa_key, Body=json.dumps(report, indent=2).encode())
    print("QA report", report)

if __name__ == "__main__":
    main()
