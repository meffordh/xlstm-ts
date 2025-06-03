"""
train_spy_models.py
Trains two xLSTM-TS checkpoints for SPY and stores them under
    weights/xlstm_spy_daily.pth   (20 epochs, 1-day bars)
    weights/xlstm_spy_hourly.pth  (15 epochs, 1-hour bars)

Each checkpoint also carries ("scaler": (mu, sigma)) so inference can
de-normalise predictions exactly.
"""
# ── std-lib & pip imports ────────────────────────────────────────────────
import os, sys, time, pathlib, datetime as dt, requests, torch, yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr

# ── repo path ────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ── repo helpers ─────────────────────────────────────────────────────────
from ml.data.preprocessing             import wavelet_denoising
from ml.models.xlstm_ts.preprocessing  import normalise_data_xlstm, create_sequences
from ml.models.xlstm_ts.preprocessing  import split_train_val_test_xlstm
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model
from ml.models.xlstm_ts.training       import train_model

np2t = lambda a: torch.tensor(a, dtype=torch.float32)

# ─────────────────────────────────────────────────────────────────────────
def tiingo_hourly(ticker: str, start: str, lookback: int = 336) -> pd.Series:
    url = f"https://api.tiingo.com/iex/{ticker}/prices"
    resp = requests.get(url, params={
        "token"       : os.environ["TIINGO_TOKEN"],
        "startDate"   : start,
        "resampleFreq": "1Hour",
        "afterHours": "true",
        "columns"     : "close"
    }, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if len(df) < lookback:
        raise RuntimeError(f"Tiingo returned {len(df)} rows (<{lookback})")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df["close"].astype(float).tail(lookback).rename("Close")

def robust_download(ticker: str, start: str, end: str,
                    interval: str = "1d", tries: int = 5) -> pd.Series:
    for _ in range(tries):
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, progress=False, threads=False)
        if not df.empty:
            return df["Close"].dropna()
        time.sleep(2)
    if interval != "1d":
        return tiingo_hourly(ticker, start)
    df = pdr.DataReader(ticker, "stooq", start, end).sort_index()
    if df.empty:
        raise RuntimeError("Stooq empty after Yahoo retries.")
    return df["Close"]

# ─────────────────────────────────────────────────────────────────────────
def make_dataset(prices: pd.Series, lookback: int):
    vals = wavelet_denoising(prices.values)
    vals, scaler = normalise_data_xlstm(vals)
    X, y, dates  = create_sequences(vals, prices.index)
    tr_end = dates.iloc[int(0.85 * len(dates))]
    vl_end = dates.iloc[int(0.925 * len(dates))]
    splits = split_train_val_test_xlstm(
        X, y, dates,
        train_end_date=tr_end, val_end_date=vl_end,
        scaler=scaler, stock="SPY"
    )
    return splits, scaler

def train_frequency(tag: str, prices: pd.Series,
                    lookback: int, epochs_hint: int):
    splits, scaler     = make_dataset(prices, lookback)
    tr_x, tr_y, vl_x, vl_y = map(np2t, (splits[0], splits[1],
                                        splits[3], splits[4]))
    model, in_proj, out_proj = create_xlstm_model(lookback)
    model, in_proj, out_proj = train_model(model, in_proj, out_proj,
                                           tr_x, tr_y, vl_x, vl_y)
    ckpt = ROOT / "weights" / f"xlstm_spy_{tag}.pth"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({
        "xlstm":  model.state_dict(),
        "in_proj": in_proj.state_dict(),
        "out_proj": out_proj.state_dict(),
        "scaler": scaler                     # ← persist (mu, sigma)
    }, ckpt)
    print(f"✅ saved {ckpt}")

# ─────────────────────────────────────────────────────────────────────────
def main(hourly_only: bool = False):
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")

    if not hourly_only:
        daily = robust_download("SPY", "2000-01-01", today, "1d")
        train_frequency("daily", daily, lookback=512, epochs_hint=20)

    end   = dt.datetime.utcnow()
    start = (end - dt.timedelta(days=365)).strftime("%Y-%m-%d")
    hourly = tiingo_hourly("SPY", start)
    train_frequency("hourly", hourly, lookback=336, epochs_hint=15)

    print("✔ all done")

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly-only", action="store_true")
    main(**vars(ap.parse_args()))
