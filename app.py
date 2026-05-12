from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd

app = Flask(
    __name__,
    template_folder='.',
    static_folder='.'
)

# Serve CSS manually for flat structure
@app.route('/style.css')
def serve_css():
    return app.send_static_file('style.css')


# ---------------- SYMBOL ---------------- #

def normalize_symbol(symbol):

    symbol = symbol.strip().upper()

    if not symbol.endswith(".NS"):
        symbol += ".NS"

    return symbol


# ---------------- FETCH DATA ---------------- #

def fetch_stock_data(symbol):

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period="3mo",
        interval="1d",
        auto_adjust=True
    )

    if df.empty:
        raise ValueError(
            f"No data found for '{symbol}'. "
            "Try TCS, RELIANCE, INFY, HDFCBANK"
        )

    return df


# ---------------- PRICE PREDICTION ---------------- #

def predict_next_price(closes):

    x = np.arange(len(closes))

    slope, intercept = np.polyfit(x, closes, 1)

    next_day = len(closes)

    predicted = slope * next_day + intercept

    return round(float(predicted), 2)


# ---------------- TREND ---------------- #

def calculate_trend(closes):

    short_ma = np.mean(closes[-5:])

    long_ma = np.mean(closes[-20:])

    if short_ma > long_ma:
        return "Bullish"
    else:
        return "Bearish"


# ---------------- MOMENTUM ---------------- #

def calculate_momentum(closes):

    momentum = closes[-1] - closes[-7]

    if momentum > 0:
        return "Strong"
    else:
        return "Weak"


# ---------------- RISK ---------------- #

def calculate_risk(closes):

    volatility = np.std(closes[-20:])

    if volatility < 20:
        return "Low"

    elif volatility < 50:
        return "Medium"

    else:
        return "High"


# ---------------- CONFIDENCE ---------------- #

def calculate_confidence(change_pct):

    confidence = 50 + abs(change_pct) * 5

    return min(round(confidence, 2), 95)


# ---------------- RSI ---------------- #

def calculate_rsi(closes, period=14):

    deltas = np.diff(closes)

    gains = np.where(deltas > 0, deltas, 0)

    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])

    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


# ---------------- PREDICTION RANGE ---------------- #

def calculate_prediction_range(predicted_price):

    lower = round(predicted_price * 0.98, 2)

    upper = round(predicted_price * 1.02, 2)

    return lower, upper


# ---------------- MAIN ROUTE ---------------- #

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        raw = request.form.get("symbol", "").strip()

        if not raw:

            error = "Please enter stock symbol."

        else:
            try:

                symbol = normalize_symbol(raw)

                df = fetch_stock_data(symbol)

                closes = df["Close"].values.astype(float)

                current_price = round(float(closes[-1]), 2)

                predicted_price = predict_next_price(closes)

                change_pct = round(
                    ((predicted_price - current_price) / current_price) * 100,
                    2
                )

                trend = calculate_trend(closes)

                momentum = calculate_momentum(closes)

                risk = calculate_risk(closes)

                confidence = calculate_confidence(change_pct)

                rsi = calculate_rsi(closes)

                lower_range, upper_range = calculate_prediction_range(
                    predicted_price
                )

                avg_volume = df["Volume"][-20:].mean()

                latest_volume = df["Volume"].iloc[-1]

                if latest_volume > avg_volume * 1.5:
                    volume_signal = "High Activity"
                else:
                    volume_signal = "Normal"

                if trend == "Bullish" and momentum == "Strong":
                    sentiment = "Positive"

                elif trend == "Bearish":
                    sentiment = "Negative"

                else:
                    sentiment = "Neutral"

                # ---------------- QUANT SCORE ---------------- #

                score = 50

                if predicted_price > current_price:
                    score += 20
                else:
                    score -= 20

                if trend == "Bullish":
                    score += 15
                else:
                    score -= 10

                if momentum == "Strong":
                    score += 10
                else:
                    score -= 5

                if risk == "Low":
                    score += 10
                elif risk == "High":
                    score -= 10

                if confidence > 80:
                    score += 15
                elif confidence > 65:
                    score += 8
                else:
                    score -= 5

                if rsi < 30:
                    score += 10
                elif rsi > 70:
                    score -= 10

                score = max(0, min(score, 100))

                # ---------------- SIGNAL ---------------- #

                if score >= 75:

                    signal = "BUY"
                    signal_class = "buy"
                    signal_icon = "↑"

                elif score >= 50:

                    signal = "HOLD"
                    signal_class = "hold"
                    signal_icon = "→"

                else:

                    signal = "SELL"
                    signal_class = "sell"
                    signal_icon = "↓"

                # ---------------- AI EXPLANATION ---------------- #

                reasons = []

                if trend == "Bullish":
                    reasons.append("Bullish trend detected")

                if momentum == "Strong":
                    reasons.append("Strong market momentum")

                if risk == "Low":
                    reasons.append("Low volatility risk")

                if confidence > 70:
                    reasons.append("High confidence score")

                if volume_signal == "High Activity":
                    reasons.append("Unusual trading activity observed")

                explanation = " • ".join(reasons)

                if explanation == "":
                    explanation = "Mixed quant market conditions detected"

                result = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "change_pct": change_pct,
                    "signal": signal,
                    "signal_class": signal_class,
                    "signal_icon": signal_icon,
                    "trend": trend,
                    "momentum": momentum,
                    "risk": risk,
                    "confidence": confidence,
                    "score": score,
                    "rsi": rsi,
                    "prediction_lower": lower_range,
                    "prediction_upper": upper_range,
                    "volume_signal": volume_signal,
                    "sentiment": sentiment,
                    "explanation": explanation
                }

            except ValueError as e:

                error = str(e)

            except Exception as e:

                error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
