import pandas as pd
from src.core.registry import Registry
from src.io.store import make_store

def main():
    reg = Registry()
    store = reg.make_store()

    assets = [a["code"] for a in reg.assets]
    frames = []
    for symbol in assets:
        key = f"{reg.silver_prefix}/daily_prices/asset={symbol}/series.parquet"
        df = store.read_parquet(key)
        df = df[["date","adj_close"]].rename(columns={"adj_close": symbol})
        frames.append(df)

    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on="date", how="inner")

    panel = panel.sort_values("date").reset_index(drop=True)
    store.put_parquet_df(f"{reg.silver_prefix}/daily_prices/panel.parquet", panel)
    print(f"wrote local {reg.silver_prefix}/daily_prices/panel.parquet rows {len(panel)}")

if __name__ == "__main__":
    main()
