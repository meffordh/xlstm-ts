# predict_direction.py
"""
Print “UP” or “DOWN” for SPY.
  MODE = "daily"   → today-vs-yesterday
  MODE = "hourly"  → next-hour drift
Run inside /workspace/xlstm-ts on the same pod that trained the weights.
"""

import os, sys, pathlib, datetime as dt, requests, pandas as pd, torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ml.data.preprocessing            import wavelet_denoising
from ml.models.xlstm_ts.preprocessing import normalise_data_xlstm, create_sequences
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model

# -- choose forecast granularity --
MODE      = "daily"           # "daily"  or "hourly"
LOOKBACK  = 256 if MODE == "daily" else 336
CKPT_FILE = ROOT / "weights" / f"xlstm_spy_{MODE}.pth"

# -- device handling (GPU if available) --
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. load checkpoint (CPU first, then .to(device))
ckpt   = torch.load(CKPT_FILE, map_location="cpu")
model, in_proj, out_proj = create_xlstm_model(LOOKBACK)   # built on CPU
model.load_state_dict(ckpt["xlstm"])
in_proj.load_state_dict(ckpt["in_proj"])
out_proj.load_state_dict(ckpt["out_proj"])

model     = model.to(device).eval()
in_proj   = in_proj.to(device)
out_proj  = out_proj.to(device)

# 2. fetch most-recent price window
start_date = (dt.datetime.utcnow() - dt.timedelta(days=LOOKBACK*3)).strftime("%Y-%m-%d")

if MODE == "daily":
    from pandas_datareader import data as pdr
    prices = (pdr.DataReader("SPY", "stooq", start_date, dt.datetime.utcnow())
                .sort_index())["Close"].astype(float)
else:
    url = "https://api.tiingo.com/iex/spy/prices"
    params = {
        "token": os.environ["TIINGO_TOKEN"],
        "startDate": start_date,
        "resampleFreq": "1Hour",
        "columns": "close"
    }
    rows = requests.get(url, params=params, timeout=30).json()
    prices = (pd.DataFrame(rows)
                .assign(date=lambda d: pd.to_datetime(d["date"]))
                .set_index("date")["close"].astype(float))

prices = prices.tail(LOOKBACK)
if len(prices) < LOOKBACK:
    raise SystemExit(f"Need {LOOKBACK} rows, got {len(prices)}")

# 3. pre-process exactly like training
series = wavelet_denoising(prices.values)
series, _ = normalise_data_xlstm(series)
X, _, _   = create_sequences(series, prices.index)
inp       = torch.tensor(X[-1:], dtype=torch.float32).to(device)   # (1,L,1)

# 4. predict & print direction
with torch.no_grad():
    forecast = out_proj(model(in_proj(inp)))[0, -1].item()

last_close = prices.iloc[-1]
direction  = "UP" if forecast > last_close else "DOWN"
print(f"{MODE.capitalize()} forecast: {forecast:.2f} | last close {last_close:.2f} → {direction}")
