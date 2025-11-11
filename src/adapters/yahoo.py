import pandas as pd
import yfinance as yf

def fetch_eod(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    df = df.reset_index().rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    })
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["date","open","high","low","close","adj_close","volume"]].sort_values("date")
    return df
