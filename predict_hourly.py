#!/usr/bin/env python3
"""
Periodically run the hourly SPY predictor and maintain a CSV with
predicted price, direction, and later the realised value.

Schedule this script to run once every hour (e.g. cron).
It appends the new prediction and updates any previous rows for which
the true close price is now available.
"""

import os
import pathlib
import datetime as dt
import pandas as pd
import torch
import requests
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ml.data.preprocessing import wavelet_denoising
from ml.models.xlstm_ts.preprocessing import create_sequences
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model

LOOKBACK = 336
CKPT_FILE = ROOT / "weights" / "xlstm_spy_hourly.pth"
CSV_FILE = ROOT / "hourly_predictions.csv"
TIINGO_TOKEN = os.environ["TIINGO_TOKEN"]  # must be set in the environment

# ---------------------------------------------------------------------------

def tiingo_hourly(ticker: str, start: str, end: str | None = None,
                  lookback: int = LOOKBACK) -> pd.Series:
    """Return the last `lookback` hourly closes for `ticker`."""
    url = f"https://api.tiingo.com/iex/{ticker}/prices"
    params = {
        "token": TIINGO_TOKEN,
        "startDate": start,
        "resampleFreq": "1Hour",
        "columns": "close",
    }
    if end:
        params["endDate"] = end
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = pd.DataFrame(r.json())
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("date", inplace=True)
    closes = data["close"].astype(float).rename("Close").tail(lookback)
    if len(closes) < lookback:
        raise RuntimeError(f"Tiingo returned {len(closes)} rows (<{lookback})")
    return closes

# ---------------------------------------------------------------------------

def load_model(device: torch.device):
    ckpt = torch.load(CKPT_FILE, map_location="cpu")
    model, in_proj, out_proj = create_xlstm_model(LOOKBACK)
    model.load_state_dict(ckpt["xlstm"])
    in_proj.load_state_dict(ckpt["in_proj"])
    out_proj.load_state_dict(ckpt["out_proj"])
    model = model.to(device).eval()
    in_proj = in_proj.to(device)
    out_proj = out_proj.to(device)
    scaler = ckpt.get("scaler")
    return model, in_proj, out_proj, scaler

def fetch_prices():
    end = dt.datetime.utcnow()
    start = (end - dt.timedelta(days=LOOKBACK * 3)).strftime("%Y-%m-%d")
    return tiingo_hourly("SPY", start)

def prepare_input(prices, scaler, device):
    series = wavelet_denoising(prices.values)

    if isinstance(scaler, tuple):
        mu, sigma = scaler
        series_norm = (series - mu) / sigma
        inv_scale = lambda z: z * sigma + mu
    else:
        series_norm = scaler.transform(series.reshape(-1, 1)).ravel()
        inv_scale = lambda z: scaler.inverse_transform([[z]])[0, 0]

    X, _, _ = create_sequences(series_norm, prices.index)
    inp = torch.tensor(X[-1:], dtype=torch.float32).to(device)
    return inp, inv_scale

def predict_next_hour(model, in_proj, out_proj, inp, inv_scale):
    with torch.no_grad():
        scaled = out_proj(model(in_proj(inp)))[0, -1].item()
    return inv_scale(scaled)

# ---------------------------------------------------------------------------

def load_csv(path: pathlib.Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame(
        columns=[
            "timestamp",
            "pred_close",
            "last_close",
            "pred_direction",
            "actual_close",
            "actual_direction",
            "correct",
        ]
    )

def update_actuals(df: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    for idx, row in df[df["actual_close"].isna()].iterrows():
        ts = row["timestamp"]
        if ts in prices.index:
            actual = float(prices.loc[ts])
            actual_dir = "UP" if actual > row["last_close"] else "DOWN"
            correct = actual_dir == row["pred_direction"]
            df.loc[idx, "actual_close"] = actual
            df.loc[idx, "actual_direction"] = actual_dir
            df.loc[idx, "correct"] = bool(correct)
    return df

def append_prediction(df: pd.DataFrame, ts: pd.Timestamp,
                      pred: float, last: float) -> pd.DataFrame:
    direction = "UP" if pred > last else "DOWN"
    new_row = pd.DataFrame(
        [{
            "timestamp": ts,
            "pred_close": pred,
            "last_close": last,
            "pred_direction": direction,
            "actual_close": pd.NA,
            "actual_direction": pd.NA,
            "correct": pd.NA,
        }]
    )
    return pd.concat([df, new_row], ignore_index=True)

# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, in_proj, out_proj, scaler = load_model(device)
    prices = fetch_prices()

    inp, inv_scale = prepare_input(prices, scaler, device)
    forecast = predict_next_hour(model, in_proj, out_proj, inp, inv_scale)

    last_close = float(prices.iloc[-1])
    next_timestamp = prices.index[-1] + pd.Timedelta(hours=1)

    df = load_csv(CSV_FILE)
    df = update_actuals(df, prices)
    df = append_prediction(df, next_timestamp, forecast, last_close)
    df.to_csv(CSV_FILE, index=False)

    print(
        f"Predicted close {forecast:.2f} for {next_timestamp} "
        f"(last close {last_close:.2f}) "
        f"→ {df.iloc[-1]['pred_direction']}"
    )

if __name__ == "__main__":
    main()
