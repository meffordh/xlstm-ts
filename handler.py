"""
handler.py – RunPod Serverless entry-point
• one endpoint handles daily (“D”) and hourly (“H”) requests
• lazy-loads the right checkpoint on first use
"""

import json, os, torch
import pandas as pd, datetime as dt, requests

ROOT = os.path.dirname(__file__)
LOOKBACK_D = 256                 # daily model
LOOKBACK_H = 336                 # hourly model
_MODELS   = {}                   # cache { "D": fn, "H": fn }

# --- helper -----------------------------------------------------------------
def load_model(freq: str):
    """Load weights → return lambda that maps (1,L,1) tensor → forecast np array"""
    ckpt_path = (f"{ROOT}/weights/"
                 f"xlstm_spy_{'daily' if freq=='D' else 'hourly'}.pth")
    ckpt = torch.load(ckpt_path,
                      map_location="cuda" if torch.cuda.is_available() else "cpu")

    from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model
    lookback = LOOKBACK_D if freq == "D" else LOOKBACK_H
    xlstm, inproj, outproj = create_xlstm_model(lookback)

    xlstm.load_state_dict(ckpt["xlstm"])
    inproj.load_state_dict(ckpt["in_proj"])
    outproj.load_state_dict(ckpt["out_proj"])
    xlstm.eval()

    def _predict(ts: torch.Tensor):
        with torch.no_grad():
            return outproj(xlstm(inproj(ts))).cpu().numpy()
    return _predict

# --- data fetch (Tiingo one-shot) -------------------------------------------
def fetch_prices(start: str, freq: str) -> pd.Series:
    url = f"https://api.tiingo.com/iex/spy/prices"
    params = {
        "token": os.environ["TIINGO_TOKEN"],
        "startDate": start,
        "resampleFreq": "1Hour" if freq == "H" else "1day",
        "columns": "close"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise RuntimeError("Tiingo returned no rows.")
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df['close'].astype(float)

# --- RunPod entry -----------------------------------------------------------
def handler(event, _context):
    """
    POST JSON
      {
        "freq": "D" | "H",         (default "D")
        "lookback": 256 | 336,     (optional)
        "start": "YYYY-MM-DD"      (optional – defaults to today-LB)
      }
    Returns {"forecast": [float, …]}
    """
    body = json.loads(event["body"])
    freq  = body.get("freq", "D").upper()
    LB    = body.get("lookback", LOOKBACK_D if freq == "D" else LOOKBACK_H)

    # 1 — load model if needed
    if freq not in _MODELS:
        _MODELS[freq] = load_model(freq)

    # 2 — fetch price window
    end = dt.datetime.utcnow().date()
    start = body.get("start",
                     (end - dt.timedelta(days=LB*3)).strftime("%Y-%m-%d"))
    prices = fetch_prices(start, freq).tail(LB)
    if len(prices) < LB:
        return {"statusCode": 400,
                "body": f"need {LB} rows, got {len(prices)}"}

    # 3 — predict
    ts = torch.tensor(prices.values, dtype=torch.float32).view(1, -1, 1)
    forecast = _MODELS[freq](ts)[0, -1].item()

    return {"statusCode": 200,
            "body": json.dumps({"forecast": forecast})}
