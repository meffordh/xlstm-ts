#!/usr/bin/env python3
"""Backtest xLSTM-TS hourly model on historic SPY data.

This script walks through the local hourly dataset contained under
`data/datasets/sp500_hourly.csv` and produces a CSV with, for each
hourly bar, the predicted close, actual close and whether the predicted
price direction was correct.
"""

import pathlib
import sys
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ml.data.preprocessing import wavelet_denoising
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model

LOOKBACK = 336
CKPT_FILE = ROOT / "weights" / "xlstm_spy_hourly.pth"
DATA_FILE = ROOT / "data" / "datasets" / "sp500_hourly.csv"
OUT_FILE = ROOT / "data" / "predictions" / "backtest_spy_hourly.csv"


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


def scale_series(series, scaler):
    if isinstance(scaler, tuple):
        mu, sigma = scaler
        norm = (series - mu) / sigma
        inv = lambda z: z * sigma + mu
    else:
        norm = scaler.transform(series.reshape(-1, 1)).ravel()
        inv = lambda z: scaler.inverse_transform([[z]])[0, 0]
    return norm, inv


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, in_proj, out_proj, scaler = load_model(device)

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    closes = df["Close"].astype(float)

    rows = []
    for idx in range(LOOKBACK, len(closes)):
        window = closes.iloc[idx - LOOKBACK:idx]
        last_close = window.iloc[-1]

        series = wavelet_denoising(window.values)
        series_norm, inv = scale_series(series, scaler)

        inp = torch.tensor(series_norm, dtype=torch.float32, device=device).view(1, -1, 1)
        with torch.no_grad():
            pred_scaled = out_proj(model(in_proj(inp)))[0, -1].item()
        pred_close = inv(pred_scaled)

        actual_close = closes.iloc[idx]
        pred_dir = "UP" if pred_close > last_close else "DOWN"
        actual_dir = "UP" if actual_close > last_close else "DOWN"

        rows.append({
            "timestamp": closes.index[idx],
            "pred_close": pred_close,
            "actual_close": actual_close,
            "last_close": last_close,
            "pred_direction": pred_dir,
            "actual_direction": actual_dir,
            "correct": pred_dir == actual_dir,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE} with {len(out_df)} rows")


if __name__ == "__main__":
    main()
