import os
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from flask import Flask, request, jsonify, send_from_directory

warnings.filterwarnings("ignore")
import yfinance as yf

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

_HERE = os.path.dirname(os.path.abspath(__file__))

# ====================== STATIC FILES ======================
@app.route("/")
def index():
    return send_from_directory(_HERE, "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(_HERE, "style.css")


@app.route("/ping")
def ping():
    return "pong", 200


# ====================== SEARCH ======================
@app.route("/search", methods=["GET"])
def search_suggestions():
    return jsonify([
        {"symbol": "RELIANCE", "name": "Reliance Industries"},
        {"symbol": "TCS", "name": "Tata Consultancy Services"},
        {"symbol": "INFY", "name": "Infosys"},
        {"symbol": "HDFCBANK", "name": "HDFC Bank"},
        {"symbol": "ICICIBANK", "name": "ICICI Bank"},
        {"symbol": "SBIN", "name": "State Bank of India"},
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance"},
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever"},
        {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank"},
        {"symbol": "TATAMOTORS", "name": "Tata Motors"},
        {"symbol": "ADANIENT", "name": "Adani Enterprises"},
        {"symbol": "AXISBANK", "name": "Axis Bank"},
        {"symbol": "MARUTI", "name": "Maruti Suzuki"},
        {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical"},
        {"symbol": "ONGC", "name": "ONGC"},
        {"symbol": "NTPC", "name": "NTPC"},
        {"symbol": "LT", "name": "Larsen & Toubro"},
        {"symbol": "POWERGRID", "name": "Power Grid"},
        {"symbol": "ULTRACEMCO", "name": "UltraTech Cement"},
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel"},
    ])


# ====================== DATA FETCH ======================
def _fetch_data(nse_symbol: str):
    try:
        ticker = yf.Ticker(nse_symbol)
        hist = ticker.history(period="6mo", interval="1d", auto_adjust=True, timeout=10)
        info = ticker.info or {}
        return hist, info
    except Exception as e:
        print(f"[ERROR] {nse_symbol}: {e}")
        return None, {}


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


# ====================== MAIN ANALYZE ======================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").upper().strip()

    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    # Clean symbol
    nse_symbol = symbol.replace(".BO", ".NS")
    if not nse_symbol.endswith(".NS"):
        nse_symbol += ".NS"

    # Fetch data with timeout
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(_fetch_data, nse_symbol)
            hist, info = future.result(timeout=12)
    except FuturesTimeoutError:
        return jsonify({"error": "Request timed out. Please try again."}), 504
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

    if hist is None or hist.empty:
        return jsonify({"error": f'No data found for "{symbol}"'}), 404

    if len(hist) < 20:
        return jsonify({"error": f'Insufficient data for "{symbol}"'}), 404

    closes = hist["Close"].values.astype(float)
    volumes = hist["Volume"].values.astype(float)
    highs = hist["High"].values.astype(float)
    lows = hist["Low"].values.astype(float)
    n = len(closes)
    current = float(closes[-1])

    # Chart Data
    last_30 = hist.tail(30)
    chart_dates = [d.strftime("%d %b") for d in last_30.index]
    chart_closes = [round(float(v), 2) for v in last_30["Close"].values]
    chart_volumes = [int(v) for v in last_30["Volume"].values]

    # Weighted Linear Regression
    x = np.arange(n, dtype=float)
    weights = np.exp(np.linspace(-1.5, 0, n))
    w_sum = weights.sum()
    x_wmean = (weights * x).sum() / w_sum
    y_wmean = (weights * closes).sum() / w_sum
    w_cov = (weights * (x - x_wmean) * (closes - y_wmean)).sum() / w_sum
    w_var = (weights * (x - x_wmean) ** 2).sum() / w_sum
    slope = w_cov / w_var if w_var != 0 else 0.0
    intercept = y_wmean - slope * x_wmean

    predicted = float(slope * n + intercept)
    fitted = slope * x + intercept
    residuals = closes - fitted
    std_res = float(np.std(residuals))

    pred_low = float(predicted - 1.5 * std_res)
    pred_high = float(predicted + 1.5 * std_res)

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((closes - np.mean(closes)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    confidence = float(max(0.0, min(100.0, r2 * 100)))

    # Moving Averages
    ma20 = float(np.mean(closes[-20:]))
    ma50 = float(np.mean(closes[-50:])) if n >= 50 else ma20

    # Trend, Momentum, RSI, Volatility, Volume logic (same as before)
    if current > ma20 and ma20 > ma50:
        trend, ts = "STRONG UPTREND", 25
    elif current > ma20:
        trend, ts = "UPTREND", 18
    elif current < ma20 and ma20 < ma50:
        trend, ts = "STRONG DOWNTREND", 0
    elif current < ma20:
        trend, ts = "DOWNTREND", 7
    else:
        trend, ts = "SIDEWAYS", 12

    mom5 = float((closes[-1] - closes[-6]) / closes[-6] * 100) if n >= 6 else 0.0
    if mom5 > 3:
        ms, msc = "STRONG BULLISH", 20
    elif mom5 > 0:
        ms, msc = "BULLISH", 15
    elif mom5 < -3:
        ms, msc = "STRONG BEARISH", 0
    else:
        ms, msc = "BEARISH", 5

    rsi = calc_rsi(closes)
    if rsi < 30:
        rs, rsc = "OVERSOLD", 20
    elif rsi > 70:
        rs, rsc = "OVERBOUGHT", 5
    elif 40 <= rsi <= 60:
        rs, rsc = "NEUTRAL", 12
    elif rsi < 40:
        rs, rsc = "WEAK", 8
    else:
        rs, rsc = "STRONG", 17

    vol_ann = float(np.std(np.diff(closes) / closes[:-1]) * np.sqrt(252) * 100)
    if vol_ann < 20:
        vl, vsc = "LOW", 15
    elif vol_ann < 40:
        vl, vsc = "MODERATE", 12
    elif vol_ann < 60:
        vl, vsc = "HIGH", 7
    else:
        vl, vsc = "VERY HIGH", 3

    avg_vol = float(np.mean(volumes[-20:])) if n >= 20 else float(np.mean(volumes))
    vol_rat = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0

    if vol_rat > 1.5:
        vsig, vasc = "HIGH ACTIVITY", 10
    elif vol_rat > 1.0:
        vsig, vasc = "ABOVE AVERAGE", 8
    elif vol_rat > 0.5:
        vsig, vasc = "BELOW AVERAGE", 5
    else:
        vsig, vasc = "LOW ACTIVITY", 3

    quant = float(min(100, ts + msc + rsc + vsc + vasc))
    chg_pct = float((predicted - current) / current * 100)

    if quant >= 65 and chg_pct > 0.5:
        rec = "BUY"
    elif quant <= 35 or chg_pct < -1.5:
        rec = "SELL"
    else:
        rec = "HOLD"

    if quant >= 70:
        sent = "STRONGLY BULLISH"
    elif quant >= 55:
        sent = "BULLISH"
    elif quant >= 45:
        sent = "NEUTRAL"
    elif quant >= 30:
        sent = "BEARISH"
    else:
        sent = "STRONGLY BEARISH"

    expl = [
        f"Trend analysis shows {trend.lower()} with price at ₹{current:.2f}.",
        f"Momentum indicates {ms.lower()} conditions.",
        f"RSI-14 is {rsi:.1f} ({rs.lower()}).",
        f"Annualized volatility is {vol_ann:.1f}% ({vl.lower()}).",
        f"Volume is {vol_rat:.1f}x average.",
        f"Regression projects ₹{predicted:.2f} (range ₹{pred_low:.2f}–₹{pred_high:.2f}).",
    ]

    # Additional Info
    cname = str(info.get("longName") or info.get("shortName") or symbol)
    sector = str(info.get("sector") or "N/A")
    w52h = float(info.get("fiftyTwoWeekHigh") or highs.max())
    w52l = float(info.get("fiftyTwoWeekLow") or lows.min())

    return jsonify({
        "symbol": nse_symbol,
        "company_name": cname,
        "sector": sector,
        "current_price": round(current, 2),
        "predicted_price": round(predicted, 2),
        "pred_low": round(pred_low, 2),
        "pred_high": round(pred_high, 2),
        "price_change_pct": round(chg_pct, 2),
        "day_change": round((current - closes[-2])/closes[-2]*100, 2) if n > 1 else 0,
        "recommendation": rec,
        "trend": trend,
        "momentum_signal": ms,
        "rsi": round(rsi, 2),
        "rsi_signal": rs,
        "volatility": round(vol_ann, 2),
        "vol_label": vl,
        "vol_signal": vsig,
        "ma20": round(ma20, 2),
        "confidence": round(confidence, 1),
        "quant_score": round(quant, 1),
        "sentiment": sent,
        "week_52_high": round(w52h, 2),
        "week_52_low": round(w52l, 2),
        "explanations": expl,
        "chart_dates": chart_dates,
        "chart_closes": chart_closes,
        "chart_volumes": chart_volumes,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
