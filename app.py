import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings('ignore')

# ── yfinance import with tz-cache in writable dir ──────────────────────────
import yfinance as yf
yf.set_tz_cache_location("/tmp")

# ── App ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ── Helpers ────────────────────────────────────────────────────────────────

def _fetch_data(nse_symbol: str):
    """Run inside a thread so we can kill it on timeout."""
    ticker = yf.Ticker(nse_symbol)
    hist = ticker.history(
        period='6mo',
        interval='1d',
        auto_adjust=True,
        timeout=8,
    )
    # info is optional – never let it block
    try:
        info = ticker.fast_info or {}
        info = dict(info)
    except Exception:
        info = {}
    # fallback to basic_info keys if fast_info missing expected keys
    if not info.get('lastPrice'):
        try:
            raw = ticker.info or {}
            info.update(raw)
        except Exception:
            pass
    return hist, info


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0,  deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'index.html'), 'r') as f:
        return f.read(), 200, {'Content-Type': 'text/html'}


@app.route('/style.css')
def serve_css():
    with open(os.path.join(os.path.dirname(__file__), 'style.css'), 'r') as f:
        return f.read(), 200, {'Content-Type': 'text/css'}


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid request body'}), 400

    symbol = data.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'error': 'Symbol is required'}), 400

    # Add .NS suffix if needed
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        nse_symbol = symbol + '.NS'
    else:
        nse_symbol = symbol

    # ── Fetch with hard 12-second wall-clock timeout ───────────────────────
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_fetch_data, nse_symbol)
            try:
                hist, info = future.result(timeout=12)
            except FuturesTimeoutError:
                return jsonify({'error': 'Data fetch timed out. Please try again.'}), 504
    except Exception as exc:
        return jsonify({'error': f'Fetch failed: {str(exc)}'}), 500

    if hist is None or hist.empty:
        return jsonify({'error': f'No data found for "{symbol}". Check ticker symbol.'}), 404

    if len(hist) < 20:
        return jsonify({'error': f'Insufficient historical data for "{symbol}".'}), 404

    # ── Arrays ─────────────────────────────────────────────────────────────
    closes  = hist['Close'].values.astype(float)
    volumes = hist['Volume'].values.astype(float)
    highs   = hist['High'].values.astype(float)
    lows    = hist['Low'].values.astype(float)
    n = len(closes)

    # ── Chart (last 30 days) ───────────────────────────────────────────────
    last_30      = hist.tail(30)
    chart_dates  = [d.strftime('%d %b') for d in last_30.index]
    chart_closes = [round(float(v), 2) for v in last_30['Close'].values]
    chart_volumes = [int(v) for v in last_30['Volume'].values]

    # ── Weighted Linear Regression ─────────────────────────────────────────
    x       = np.arange(n, dtype=float)
    weights = np.exp(np.linspace(-1.5, 0, n))
    w_sum   = weights.sum()
    x_wmean = (weights * x).sum() / w_sum
    y_wmean = (weights * closes).sum() / w_sum
    w_cov   = (weights * (x - x_wmean) * (closes - y_wmean)).sum() / w_sum
    w_var   = (weights * (x - x_wmean) ** 2).sum() / w_sum
    slope   = w_cov / w_var if w_var != 0 else 0.0
    intercept = y_wmean - slope * x_wmean
    next_x    = float(n)
    predicted_price = float(slope * next_x + intercept)
    fitted    = slope * x + intercept
    residuals = closes - fitted
    std_res   = float(np.std(residuals))
    pred_low  = float(predicted_price - 1.5 * std_res)
    pred_high = float(predicted_price + 1.5 * std_res)
    current_price = float(closes[-1])

    # R² / confidence
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((closes - np.mean(closes)) ** 2))
    r_squared  = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    confidence = float(max(0.0, min(100.0, r_squared * 100)))

    # ── Moving Averages ────────────────────────────────────────────────────
    ma20 = float(np.mean(closes[-20:]))
    ma50 = float(np.mean(closes[-50:])) if n >= 50 else float(np.mean(closes))

    # ── Trend ──────────────────────────────────────────────────────────────
    if   current_price > ma20 and ma20 > ma50: trend, trend_score = 'STRONG UPTREND',   25
    elif current_price > ma20:                  trend, trend_score = 'UPTREND',           18
    elif current_price < ma20 and ma20 < ma50: trend, trend_score = 'STRONG DOWNTREND',  0
    elif current_price < ma20:                  trend, trend_score = 'DOWNTREND',          7
    else:                                        trend, trend_score = 'SIDEWAYS',          12

    # ── Momentum ───────────────────────────────────────────────────────────
    mom5  = float((closes[-1] - closes[-6])  / closes[-6]  * 100) if n >= 6  else 0.0
    mom20 = float((closes[-1] - closes[-21]) / closes[-21] * 100) if n >= 21 else 0.0

    if   mom5 > 3:  momentum_signal, momentum_score = 'STRONG BULLISH', 20
    elif mom5 > 0:  momentum_signal, momentum_score = 'BULLISH',         15
    elif mom5 < -3: momentum_signal, momentum_score = 'STRONG BEARISH',  0
    else:           momentum_signal, momentum_score = 'BEARISH',          5

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi = calc_rsi(closes)
    if   rsi < 30:          rsi_signal, rsi_score = 'OVERSOLD',   20
    elif rsi > 70:          rsi_signal, rsi_score = 'OVERBOUGHT',  5
    elif 40 <= rsi <= 60:   rsi_signal, rsi_score = 'NEUTRAL',    12
    elif rsi < 40:          rsi_signal, rsi_score = 'WEAK',        8
    else:                   rsi_signal, rsi_score = 'STRONG',     17

    # ── Volatility ─────────────────────────────────────────────────────────
    daily_returns = np.diff(closes) / closes[:-1]
    volatility    = float(np.std(daily_returns) * np.sqrt(252) * 100)
    if   volatility < 20: vol_label, vol_score = 'LOW',       15
    elif volatility < 40: vol_label, vol_score = 'MODERATE',  12
    elif volatility < 60: vol_label, vol_score = 'HIGH',       7
    else:                 vol_label, vol_score = 'VERY HIGH',  3

    # ── Volume ─────────────────────────────────────────────────────────────
    avg_vol20    = float(np.mean(volumes[-20:])) if n >= 20 else float(np.mean(volumes))
    current_vol  = float(volumes[-1])
    vol_ratio    = float(current_vol / avg_vol20) if avg_vol20 > 0 else 1.0
    if   vol_ratio > 1.5: vol_signal, vol_activity_score = 'HIGH ACTIVITY',   10
    elif vol_ratio > 1.0: vol_signal, vol_activity_score = 'ABOVE AVERAGE',    8
    elif vol_ratio > 0.5: vol_signal, vol_activity_score = 'BELOW AVERAGE',    5
    else:                 vol_signal, vol_activity_score = 'LOW ACTIVITY',      3

    # ── Quant Score & Signals ──────────────────────────────────────────────
    quant_score = float(min(100,
        trend_score + momentum_score + rsi_score + vol_score + vol_activity_score))

    price_change_pct = float((predicted_price - current_price) / current_price * 100)

    if   quant_score >= 65 and price_change_pct > 0.5:  recommendation = 'BUY'
    elif quant_score <= 35 or  price_change_pct < -1.5:  recommendation = 'SELL'
    else:                                                  recommendation = 'HOLD'

    if   quant_score >= 70: sentiment = 'STRONGLY BULLISH'
    elif quant_score >= 55: sentiment = 'BULLISH'
    elif quant_score >= 45: sentiment = 'NEUTRAL'
    elif quant_score >= 30: sentiment = 'BEARISH'
    else:                   sentiment = 'STRONGLY BEARISH'

    # ── Explanations ───────────────────────────────────────────────────────
    explanations = [
        f"Trend analysis shows {trend.lower()} structure with price at ₹{current_price:.2f}.",
        f"Momentum readings indicate {momentum_signal.lower()} conditions across short and medium timeframes.",
        f"RSI-14 currently stands at {rsi:.1f}, classified as {rsi_signal.lower()}.",
        f"Annualized volatility is {volatility:.1f}% indicating {vol_label.lower()} risk.",
        f"Volume participation is {vol_ratio:.1f}x relative to rolling 20-day average.",
        f"Statistical regression projects ₹{predicted_price:.2f} with expected range ₹{pred_low:.2f}–₹{pred_high:.2f}.",
    ]

    # ── Extra info from yfinance ───────────────────────────────────────────
    def _safe(key, fallback):
        v = info.get(key)
        if v is None or v == 'N/A':
            return fallback
        try:
            return float(v)
        except Exception:
            return fallback

    week52_high  = _safe('fiftyTwoWeekHigh',  float(highs.max()))
    week52_low   = _safe('fiftyTwoWeekLow',   float(lows.min()))
    prev_close   = _safe('previousClose',     float(closes[-2]) if n > 1 else float(closes[-1]))
    day_change   = float((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

    company_name = str(info.get('longName') or info.get('shortName') or nse_symbol)
    sector       = str(info.get('sector') or 'N/A')

    mkt_cap_raw  = info.get('marketCap') or 0
    try:
        mkt_cap_raw = float(mkt_cap_raw)
    except Exception:
        mkt_cap_raw = 0

    if   mkt_cap_raw >= 1e12: market_cap_str = f'₹{mkt_cap_raw/1e12:.2f}T'
    elif mkt_cap_raw >= 1e9:  market_cap_str = f'₹{mkt_cap_raw/1e9:.2f}B'
    elif mkt_cap_raw >= 1e6:  market_cap_str = f'₹{mkt_cap_raw/1e6:.2f}M'
    else:                      market_cap_str = 'N/A'

    week52_range = week52_high - week52_low
    week52_pos   = float((current_price - week52_low) / week52_range * 100) if week52_range > 0 else 50.0

    return jsonify({
        'symbol':           nse_symbol,
        'company_name':     company_name,
        'sector':           sector,
        'market_cap':       market_cap_str,
        'current_price':    round(current_price,     2),
        'predicted_price':  round(predicted_price,   2),
        'pred_low':         round(pred_low,           2),
        'pred_high':        round(pred_high,          2),
        'price_change_pct': round(price_change_pct,  2),
        'day_change':       round(day_change,         2),
        'recommendation':   recommendation,
        'trend':            trend,
        'momentum_signal':  momentum_signal,
        'momentum_5':       round(mom5,               2),
        'momentum_20':      round(mom20,              2),
        'rsi':              round(rsi,                2),
        'rsi_signal':       rsi_signal,
        'volatility':       round(volatility,         2),
        'vol_label':        vol_label,
        'vol_ratio':        round(vol_ratio,          2),
        'vol_signal':       vol_signal,
        'ma20':             round(ma20,               2),
        'ma50':             round(ma50,               2),
        'confidence':       round(confidence,         2),
        'quant_score':      round(quant_score,        2),
        'sentiment':        sentiment,
        'week_52_high':     round(week52_high,        2),
        'week_52_low':      round(week52_low,         2),
        'week_52_pos':      round(week52_pos,         2),
        'explanations':     explanations,
        'chart_dates':      chart_dates,
        'chart_closes':     chart_closes,
        'chart_volumes':    chart_volumes,
        'avg_volume':       round(avg_vol20,           0),
    })


@app.route('/search', methods=['GET'])
def search_suggestions():
    popular = [
        {'symbol': 'RELIANCE',   'name': 'Reliance Industries'},
        {'symbol': 'TCS',        'name': 'Tata Consultancy Services'},
        {'symbol': 'INFY',       'name': 'Infosys'},
        {'symbol': 'HDFCBANK',   'name': 'HDFC Bank'},
        {'symbol': 'ICICIBANK',  'name': 'ICICI Bank'},
        {'symbol': 'WIPRO',      'name': 'Wipro'},
        {'symbol': 'SBIN',       'name': 'State Bank of India'},
        {'symbol': 'BAJFINANCE', 'name': 'Bajaj Finance'},
        {'symbol': 'HINDUNILVR', 'name': 'Hindustan Unilever'},
        {'symbol': 'KOTAKBANK',  'name': 'Kotak Mahindra Bank'},
        {'symbol': 'TATAMOTORS', 'name': 'Tata Motors'},
        {'symbol': 'ADANIENT',   'name': 'Adani Enterprises'},
        {'symbol': 'AXISBANK',   'name': 'Axis Bank'},
        {'symbol': 'MARUTI',     'name': 'Maruti Suzuki'},
        {'symbol': 'SUNPHARMA',  'name': 'Sun Pharmaceutical'},
        {'symbol': 'ONGC',       'name': 'ONGC'},
        {'symbol': 'NTPC',       'name': 'NTPC'},
        {'symbol': 'LT',         'name': 'Larsen & Toubro'},
        {'symbol': 'POWERGRID',  'name': 'Power Grid Corporation'},
        {'symbol': 'ULTRACEMCO', 'name': 'UltraTech Cement'},
    ]
    return jsonify(popular)


# ── Health-check (keeps Render from cold-starting mid-request) ─────────────
@app.route('/ping')
def ping():
    return 'pong', 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
