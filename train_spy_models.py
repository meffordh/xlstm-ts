"""
train_spy_models.py
Trains two xLSTM-TS checkpoints for SPY and stores them under:
    weights/xlstm_spy_daily.pth    (20 epochs, 1-day bars)
    weights/xlstm_spy_hourly.pth   (15 epochs, 1-hour bars)
"""

# ── add src/ to PYTHONPATH ───────────────────────────────────────────────
import sys, pathlib, time, datetime as dt, torch, yfinance as yf, pandas as pd
from pandas_datareader import data as pdr                    # Stooq fallback

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ── add another helper for numpy to torch sensor
import torch
def np2tensor(a):                       # ← ADD
    return torch.tensor(a, dtype=torch.float32)

# ── repo helpers ─────────────────────────────────────────────────────────
from ml.data.preprocessing               import wavelet_denoising
from ml.models.xlstm_ts.preprocessing    import (
    normalise_data_xlstm,
    create_sequences,
    split_train_val_test_xlstm,
)
from ml.models.xlstm_ts.xlstm_ts_model   import create_xlstm_model
from ml.models.xlstm_ts.training         import train_model
from ml.constants                        import SEQ_LENGTH_XLSTM

# ------------------------------------------------------------------
#  robust Tiingo intraday (SPY works)
# ------------------------------------------------------------------
import os, requests, pandas as pd, datetime as dt, time

# --- ONE-SHOT Tiingo intraday -------------------------------------------
def tiingo_hourly(ticker: str,
                  start: str,
                  end: str | None = None,          # ← end is optional
                  lookback: int = 336) -> pd.Series:
    """
    Returns the last `lookback` 1-hour closes.
    If `end` is None, Tiingo returns data up to the latest bar.
    Mirrors the working browser call.
    """
    url = f"https://api.tiingo.com/iex/{ticker}/prices"
    params = {
        "token": os.environ["TIINGO_TOKEN"],
        "startDate": start,
        "resampleFreq": "1Hour",
        "columns": "close"
    }
    if end:
        params["endDate"] = end                 # include only if provided

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if len(data) < lookback:
        raise RuntimeError(f"Tiingo returned {len(data)} rows (<{lookback})")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df["close"].astype(float).rename("Close").tail(lookback)


# ── robust price downloader (Yahoo → Stooq fallback) ─────────────────────
def robust_download(ticker:str, start:str, end:str, interval:str="1d",
                    tries:int=5, sleep:int=3) -> pd.Series:
    """Return pd.Series of Close prices; retries Yahoo then falls back to Stooq."""
    for n in range(tries):
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, progress=False, threads=False)
        if not df.empty:
            return df["Close"].dropna()
        print(f"⚠ Yahoo empty {n+1}/{tries}; retrying …")
        time.sleep(sleep)

    # -------- intraday branch ----------------
    if interval != "1d":            # e.g. "1h"
        return tiingo_hourly(ticker, start)   # ← pass only start
    # -----------------------------------------

    # --------- daily branch (Stooq fallback) ---------------------------
    print("🛑 Yahoo failed — switching to Stooq")
    df = pdr.DataReader(ticker, "stooq", start, end)
    if df.empty:
        raise RuntimeError("Stooq also returned empty.")
    df = df.sort_index()                     # Stooq comes reverse-ordered
    return df["Close"].rename("Close")
    # -------------------------------------------------------------------


# ── dataset builder (keeps dates separate) ───────────────────────────────
def make_dataset(prices, lookback):
    dates = prices.index                                 # keep DatetimeIndex
    values = wavelet_denoising(prices.values)            # ndarray
    values, scaler = normalise_data_xlstm(values)
    X, y, seq_dates = create_sequences(values, dates)    # uses index we saved
    tr_end = seq_dates.iloc[int(0.85 * len(seq_dates))]
    vl_end = seq_dates.iloc[int(0.925 * len(seq_dates))]

    return split_train_val_test_xlstm(
        X, y, seq_dates,
        train_end_date=tr_end,
        val_end_date=vl_end,
        scaler=scaler,
        stock="SPY"
    )

# ── single-frequency trainer ─────────────────────────────────────────────
def train_frequency(tag, prices, epochs, lookback):
    if prices.empty:
        raise RuntimeError(f"[{tag}] price series empty after download.")

    # ------------------------------------------------------------------
    # get *everything* the helper hands back
    splits = make_dataset(prices, lookback)
    
    # correct mapping: 0 1 3 4 6 7 are the tensors, 2 5 8 are date arrays
    tr_x, tr_y = splits[0], splits[1]
    vl_x, vl_y = splits[3], splits[4]
    ts_x, ts_y = splits[6], splits[7]    # (not used, but here if you need)

    # --- NEW: NumPy → torch -------------------------------------------------
    tr_x, tr_y = np2tensor(tr_x), np2tensor(tr_y)
    vl_x, vl_y = np2tensor(vl_x), np2tensor(vl_y)
    # ------------------------------------------------------------------------

    xlstm, inp_proj, out_proj = create_xlstm_model(lookback)
    xlstm, inp_proj, out_proj = train_model(
        xlstm, inp_proj, out_proj,
        tr_x, tr_y, vl_x, vl_y,
    )

    out = ROOT / "weights" / f"xlstm_spy_{tag}.pth"
    out.parent.mkdir(exist_ok=True)
    torch.save(
        {"xlstm": xlstm.state_dict(),
         "in_proj": inp_proj.state_dict(),
         "out_proj": out_proj.state_dict()},
        out,
    )
    print(f"✅ saved {out}")


# ── main driver ──────────────────────────────────────────────────────────
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly-only", action="store_true")
    args = parser.parse_args()

    TODAY = dt.datetime.utcnow().strftime("%Y-%m-%d")

    if not args.hourly_only:
        daily = robust_download("SPY", "2000-01-01", TODAY, interval="1d")
        train_frequency("daily", daily, epochs=20, lookback=256)

    # -- main driver -------------------------------------------------
    end   = dt.datetime.utcnow()
    start = (end - dt.timedelta(days=120)).strftime("%Y-%m-%d")   # ← 120 days instead of 750
    hourly = tiingo_hourly("SPY", start)                          # no endDate needed
    train_frequency("hourly", hourly, epochs=15, lookback=336)

    print("✔ all done")
