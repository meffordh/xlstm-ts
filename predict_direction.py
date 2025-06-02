# predict_direction.py
"""
Print “UP” or “DOWN” for SPY.

• MODE = "daily"  → next-day move (Stooq, 256-bar look-back)
• MODE = "hourly" → next-hour move (Tiingo 1-hour bars, 336-bar look-back)

Requires checkpoints produced by train_spy_models.py that include:
    xlstm, in_proj, out_proj, scaler
(where scaler is either (mu, sigma) or an sklearn MinMaxScaler).
"""

import os
import sys
import pathlib
import datetime as dt
import requests
import pandas as pd
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ml.data.preprocessing import wavelet_denoising
from ml.models.xlstm_ts.preprocessing import create_sequences
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model

# ── configuration ────────────────────────────────────────────────────────
MODE = "hourly"                    # "daily" or "hourly"
LOOKBACK = 256 if MODE == "daily" else 336
CKPT_FILE = ROOT / "weights" / f"xlstm_spy_{MODE}.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. load checkpoint (weights + scaler) ───────────────────────────────
state = torch.load(CKPT_FILE, map_location="cpu")
scaler = state["scaler"]                        # (mu, σ) tuple or sklearn scaler

model, in_proj, out_proj = create_xlstm_model(LOOKBACK)
model.load_state_dict(state["xlstm"])
in_proj.load_state_dict(state["in_proj"])
out_proj.load_state_dict(state["out_proj"])

model = model.to(DEVICE).eval()
in_proj = in_proj.to(DEVICE)
out_proj = out_proj.to(DEVICE)

# ── 2. fetch most-recent LOOKBACK bars ──────────────────────────────────
start_date = (
    dt.datetime.utcnow() - dt.timedelta(days=LOOKBACK * 3)
).strftime("%Y-%m-%d")

if MODE == "daily":
    from pandas_datareader import data as pdr

    prices = (
        pdr.DataReader("SPY", "stooq", start_date, dt.datetime.utcnow())
        .sort_index()
    )["Close"].astype(float)
else:
    url = "https://api.tiingo.com/iex/spy/prices"
    rows = requests.get(
        url,
        params={
            "token": os.environ["TIINGO_TOKEN"],
            "startDate": start_date,
            "resampleFreq": "1Hour",
            "columns": "close",
        },
        timeout=30,
    ).json()
    prices = (
        pd.DataFrame(rows)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")["close"]
        .astype(float)
    )

prices = prices.tail(LOOKBACK)
if len(prices) < LOOKBACK:
    sys.exit(f"Need {LOOKBACK} rows, got {len(prices)}")

# ── 3. apply SAME scaling used in training ──────────────────────────────
series = wavelet_denoising(prices.values)

if isinstance(scaler, tuple):  # (mu, σ)
    mu, sigma = scaler
    series_norm = (series - mu) / sigma
    inv_scale = lambda z: z * sigma + mu
else:  # MinMaxScaler
    series_norm = scaler.transform(series.reshape(-1, 1)).ravel()
    inv_scale = lambda z: scaler.inverse_transform([[z]])[0, 0]

X, _, _ = create_sequences(series_norm, prices.index)
# ...  (imports and earlier code unchanged) ...

# ---- ensure shape (N, seq_len, 1)  *feature channel last* -------------
if X.ndim == 2:                       # (N, seq_len)
    X = X[:, :, None]                 # → (N, seq_len, 1)
elif X.shape[-1] != 1:                # feature not last
    # (N, 1, seq_len)  →  (N, seq_len, 1)
    X = np.transpose(X, (0, 2, 1))
# -----------------------------------------------------------------------

inp = torch.tensor(X[-1:], dtype=torch.float32, device=DEVICE)

with torch.no_grad():
    pred_scaled = out_proj(model(in_proj(inp)))[0, -1].item()

pred_raw     = inv_scale(pred_scaled)
prior_close  = prices.iloc[-1]
direction    = "UP" if pred_raw > prior_close else "DOWN"

print(f"{MODE.capitalize()} · Prior close: {prior_close:.2f} | "
      f"Predicted next: {pred_raw:.2f} | Direction: {direction}")
