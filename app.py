from flask import Flask, render_template, request
import yfinance as yf
import numpy as np

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

def fetch_closes(symbol):

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

    return df["Close"].values.astype(float)


# ---------------- PRICE PREDICTION ---------------- #

def predict_next_price(closes):

    x = np.arange(len(closes))

    slope, intercept = np.polyfit(x, closes, 1)

    next_day = len(closes)

    predicted = slope * next_day + intercept

    return round(float(predicted), 2)


# ---------------- QUANT SIGNALS ---------------- #

def calculate_trend(closes):

    short_ma = np.mean(closes[-5:])

    long_ma = np.mean(closes[-20:])

    if short_ma > long_ma:
        return "Bullish"
    else:
        return "Bearish"


def calculate_momentum(closes):

    momentum = closes[-1] - closes[-7]

    if momentum > 0:
        return "Strong"
    else:
        return "Weak"


def calculate_risk(closes):

    volatility = np.std(closes[-20:])

    if volatility < 20:
        return "Low"
    elif volatility < 50:
        return "Medium"
    else:
        return "High"


def calculate_confidence(change_pct):

    confidence = 50 + abs(change_pct) * 5

    return min(round(confidence, 2), 95)


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

                closes = fetch_closes(symbol)

                current_price = round(float(closes[-1]), 2)

                predicted_price = predict_next_price(closes)

                change_pct = round(
                    ((predicted_price - current_price) / current_price) * 100,
                    2
                )

                # -------- QUANT SIGNALS -------- #

                trend = calculate_trend(closes)

                momentum = calculate_momentum(closes)

                risk = calculate_risk(closes)

                confidence = calculate_confidence(change_pct)

                # -------- IMPROVED QUANT SCORING -------- #

                score = 50

                # Prediction direction
                if predicted_price > current_price:
                    score += 20
                else:
                    score -= 20

                # Trend strength
                if trend == "Bullish":
                    score += 15
                else:
                    score -= 10

                # Momentum
                if momentum == "Strong":
                    score += 10
                else:
                    score -= 5

                # Risk analysis
                if risk == "Low":
                    score += 10
                elif risk == "Medium":
                    score += 0
                else:
                    score -= 10

                # Confidence
                if confidence > 80:
                    score += 15
                elif confidence > 65:
                    score += 8
                else:
                    score -= 5

                # Clamp score
                score = max(0, min(score, 100))

                # -------- FINAL RECOMMENDATION -------- #

                if score >= 75:

                    signal = "BUY"
                    signal_class = "buy"
                    signal_icon = "↑"

                    explanation = (
                        "Strong bullish trend with favorable quant signals."
                    )

                elif score >= 50:

                    signal = "HOLD"
                    signal_class = "hold"
                    signal_icon = "→"

                    explanation = (
                        "Mixed market conditions with moderate confidence."
                    )

                else:

                    signal = "SELL"
                    signal_class = "sell"
                    signal_icon = "↓"

                    explanation = (
                        "Weak momentum and higher downside risk detected."
                    )

                # -------- RESULT -------- #

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
                    "explanation": explanation
                }

            except ValueError as e:

                error = str(e)

            except Exception:

                error = (
                    "Yahoo Finance rate limit reached. "
                    "Try again after a minute."
                )

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run()
