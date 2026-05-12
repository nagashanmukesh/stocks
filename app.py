import os
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
from flask import Flask, request, jsonify

warnings.filterwarnings("ignore")

import yfinance as yf
yf.set_tz_cache_location("/tmp")

# ── App ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    with open(os.path.join(_HERE, name), "r", encoding="utf-8") as f:
        return f.read()


# ── Data fetch (runs in thread for hard wall-clock timeout) ───────────────
def _fetch_data(nse_symbol: str):
    ticker = yf.Ticker(nse_symbol)
    hist = ticker.history(
        period="6mo",
        interval="1d",
        auto_adjust=True,
        timeout=8,
    )
    info = {}
    try:
        fi = ticker.fast_info
        info = {k: getattr(fi, k, None) for k in dir(fi) if not k.startswith("_")}
    except Exception:
        pass
    if not info.get("lastPrice"):
        try:
            info.update(ticker.info or {})
        except Exception:
            pass
    return hist, info


# ── RSI ────────────────────────────────────────────────────────────────────
def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas   = np.diff(prices)
    gains    = np.where(deltas > 0,  deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return _read("index.html"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/style.css")
def serve_css():
    return _read("style.css"), 200, {"Content-Type": "text/css; charset=utf-8"}


@app.route("/ping")
def ping():
    return "pong", 200


@app.route("/search", methods=["GET"])
def search_suggestions():
    return jsonify([
        {"symbol": "RELIANCE",   "name": "Reliance Industries"},
        {"symbol": "TCS",        "name": "Tata Consultancy Services"},
        {"symbol": "INFY",       "name": "Infosys"},
        {"symbol": "HDFCBANK",   "name": "HDFC Bank"},
        {"symbol": "ICICIBANK",  "name": "ICICI Bank"},
        {"symbol": "WIPRO",      "name": "Wipro"},
        {"symbol": "SBIN",       "name": "State Bank of India"},
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance"},
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever"},
        {"symbol": "KOTAKBANK",  "name": "Kotak Mahindra Bank"},
        {"symbol": "TATAMOTORS", "name": "Tata Motors"},
        {"symbol": "ADANIENT",   "name": "Adani Enterprises"},
        {"symbol": "AXISBANK",   "name": "Axis Bank"},
        {"symbol": "MARUTI",     "name": "Maruti Suzuki"},
        {"symbol": "SUNPHARMA",  "name": "Sun Pharmaceutical"},
        {"symbol": "ONGC",       "name": "ONGC"},
        {"symbol": "NTPC",       "name": "NTPC"},
        {"symbol": "LT",         "name": "Larsen & Toubro"},
        {"symbol": "POWERGRID",  "name": "Power Grid Corporation"},
        {"symbol": "ULTRACEMCO", "name": "UltraTech Cement"},
    ])


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    symbol = data.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        nse_symbol = symbol + ".NS"
    else:
        nse_symbol = symbol

    # ── Hard 12-second timeout around yfinance ────────────────────────────
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_fetch_data, nse_symbol)
            try:
                hist, info = future.result(timeout=12)
            except FuturesTimeoutError:
                return jsonify({"error": "Data fetch timed out. Please try again."}), 504
    except Exception as exc:
        return jsonify({"error": f"Fetch failed: {str(exc)}"}), 500

    if hist is None or hist.empty:
        return jsonify({"error": f'No data found for "{symbol}". Check ticker symbol.'}), 404
    if len(hist) < 20:
        return jsonify({"error": f'Insufficient historical data for "{symbol}".'}), 404

    closes  = hist["Close"].values.astype(float)
    volumes = hist["Volume"].values.astype(float)
    highs   = hist["High"].values.astype(float)
    lows    = hist["Low"].values.astype(float)
    n       = len(closes)

    # Chart
    last_30       = hist.tail(30)
    chart_dates   = [d.strftime("%d %b") for d in last_30.index]
    chart_closes  = [round(float(v), 2) for v in last_30["Close"].values]
    chart_volumes = [int(v) for v in last_30["Volume"].values]

    # Weighted linear regression
    x         = np.arange(n, dtype=float)
    weights   = np.exp(np.linspace(-1.5, 0, n))
    w_sum     = weights.sum()
    x_wmean   = (weights * x).sum() / w_sum
    y_wmean   = (weights * closes).sum() / w_sum
    w_cov     = (weights * (x - x_wmean) * (closes - y_wmean)).sum() / w_sum
    w_var     = (weights * (x - x_wmean) ** 2).sum() / w_sum
    slope     = w_cov / w_var if w_var != 0 else 0.0
    intercept = y_wmean - slope * x_wmean
    predicted = float(slope * n + intercept)
    fitted    = slope * x + intercept
    residuals = closes - fitted
    std_res   = float(np.std(residuals))
    pred_low  = float(predicted - 1.5 * std_res)
    pred_high = float(predicted + 1.5 * std_res)
    current   = float(closes[-1])

    ss_res     = float(np.sum(residuals ** 2))
    ss_tot     = float(np.sum((closes - np.mean(closes)) ** 2))
    r2         = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    confidence = float(max(0.0, min(100.0, r2 * 100)))

    ma20 = float(np.mean(closes[-20:]))
    ma50 = float(np.mean(closes[-50:])) if n >= 50 else float(np.mean(closes))

    # Trend
    if   current > ma20 and ma20 > ma50: trend, ts = "STRONG UPTREND",  25
    elif current > ma20:                  trend, ts = "UPTREND",          18
    elif current < ma20 and ma20 < ma50: trend, ts = "STRONG DOWNTREND",  0
    elif current < ma20:                  trend, ts = "DOWNTREND",          7
    else:                                  trend, ts = "SIDEWAYS",          12

    # Momentum
    mom5  = float((closes[-1] - closes[-6])  / closes[-6]  * 100) if n >= 6  else 0.0
    mom20 = float((closes[-1] - closes[-21]) / closes[-21] * 100) if n >= 21 else 0.0
    if   mom5 > 3:  ms, msc = "STRONG BULLISH", 20
    elif mom5 > 0:  ms, msc = "BULLISH",         15
    elif mom5 < -3: ms, msc = "STRONG BEARISH",  0
    else:           ms, msc = "BEARISH",          5

    # RSI
    rsi = calc_rsi(closes)
    if   rsi < 30:        rs, rsc = "OVERSOLD",   20
    elif rsi > 70:        rs, rsc = "OVERBOUGHT",  5
    elif 40 <= rsi <= 60: rs, rsc = "NEUTRAL",    12
    elif rsi < 40:        rs, rsc = "WEAK",        8
    else:                 rs, rsc = "STRONG",      17

    # Volatility
    vol_ann = float(np.std(np.diff(closes) / closes[:-1]) * np.sqrt(252) * 100)
    if   vol_ann < 20: vl, vsc = "LOW",      15
    elif vol_ann < 40: vl, vsc = "MODERATE", 12
    elif vol_ann < 60: vl, vsc = "HIGH",      7
    else:              vl, vsc = "VERY HIGH",  3

    # Volume
    avg_vol  = float(np.mean(volumes[-20:])) if n >= 20 else float(np.mean(volumes))
    vol_rat  = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0
    if   vol_rat > 1.5: vsig, vasc = "HIGH ACTIVITY",  10
    elif vol_rat > 1.0: vsig, vasc = "ABOVE AVERAGE",   8
    elif vol_rat > 0.5: vsig, vasc = "BELOW AVERAGE",   5
    else:               vsig, vasc = "LOW ACTIVITY",     3

    quant = float(min(100, ts + msc + rsc + vsc + vasc))
    chg_pct = float((predicted - current) / current * 100)

    if   quant >= 65 and chg_pct > 0.5:  rec = "BUY"
    elif quant <= 35 or  chg_pct < -1.5:  rec = "SELL"
    else:                                   rec = "HOLD"

    if   quant >= 70: sent = "STRONGLY BULLISH"
    elif quant >= 55: sent = "BULLISH"
    elif quant >= 45: sent = "NEUTRAL"
    elif quant >= 30: sent = "BEARISH"
    else:             sent = "STRONGLY BEARISH"

    expl = [
        f"Trend analysis shows {trend.lower()} structure with price at ₹{current:.2f}.",
        f"Momentum readings indicate {ms.lower()} conditions across short and medium timeframes.",
        f"RSI-14 currently stands at {rsi:.1f}, classified as {rs.lower()}.",
        f"Annualized volatility is {vol_ann:.1f}% indicating {vl.lower()} risk.",
        f"Volume participation is {vol_rat:.1f}x relative to rolling 20-day average.",
        f"Statistical regression projects ₹{predicted:.2f} with expected range ₹{pred_low:.2f}–₹{pred_high:.2f}.",
    ]

    def _safe(key, fallback):
        v = info.get(key)
        if v is None or v == "N/A":
            return fallback
        try:
            return float(v)
        except Exception:
            return fallback

    w52h = _safe("fiftyTwoWeekHigh", float(highs.max()))
    w52l = _safe("fiftyTwoWeekLow",  float(lows.min()))
    prev = _safe("previousClose",    float(closes[-2]) if n > 1 else float(closes[-1]))
    dchg = float((current - prev) / prev * 100) if prev > 0 else 0.0

    cname  = str(info.get("longName") or info.get("shortName") or nse_symbol)
    sector = str(info.get("sector") or "N/A")

    try:
        mc_raw = float(info.get("marketCap") or 0)
    except Exception:
        mc_raw = 0.0

    if   mc_raw >= 1e12: mc_str = f"₹{mc_raw/1e12:.2f}T"
    elif mc_raw >= 1e9:  mc_str = f"₹{mc_raw/1e9:.2f}B"
    elif mc_raw >= 1e6:  mc_str = f"₹{mc_raw/1e6:.2f}M"
    else:                mc_str = "N/A"

    w52r = w52h - w52l
    w52p = float((current - w52l) / w52r * 100) if w52r > 0 else 50.0

    return jsonify({
        "symbol":           nse_symbol,
        "company_name":     cname,
        "sector":           sector,
        "market_cap":       mc_str,
        "current_price":    round(current,  2),
        "predicted_price":  round(predicted, 2),
        "pred_low":         round(pred_low,  2),
        "pred_high":        round(pred_high, 2),
        "price_change_pct": round(chg_pct,  2),
        "day_change":       round(dchg,      2),
        "recommendation":   rec,
        "trend":            trend,
        "momentum_signal":  ms,
        "momentum_5":       round(mom5,     2),
        "momentum_20":      round(mom20,    2),
        "rsi":              round(rsi,      2),
        "rsi_signal":       rs,
        "volatility":       round(vol_ann,  2),
        "vol_label":        vl,
        "vol_ratio":        round(vol_rat,  2),
        "vol_signal":       vsig,
        "ma20":             round(ma20,     2),
        "ma50":             round(ma50,     2),
        "confidence":       round(confidence, 2),
        "quant_score":      round(quant,    2),
        "sentiment":        sent,
        "week_52_high":     round(w52h,     2),
        "week_52_low":      round(w52l,     2),
        "week_52_pos":      round(w52p,     2),
        "explanations":     expl,
        "chart_dates":      chart_dates,
        "chart_closes":     chart_closes,
        "chart_volumes":    chart_volumes,
        "avg_volume":       round(avg_vol,   0),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
