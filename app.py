# app.py — NSE Stock Predictor (NumPy-only, flat structure)
# Render-compatible · No scikit-learn · Beginner friendly

from flask import Flask, render_template, request
import yfinance as yf
import numpy as np

# ── Flat structure: index.html and style.css live next to app.py ──
app = Flask(
    __name__,
    template_folder='.',   # look for templates in the root folder
    static_folder='.'      # look for static files in the root folder
)


def normalize_symbol(symbol):
    """Uppercase and append .NS if the user left it off."""
    symbol = symbol.strip().upper()
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    return symbol


def fetch_closes(symbol):
    """
    Download the last 90 days of daily closing prices for an NSE stock.
    Returns a NumPy array of floats, or raises ValueError on bad symbol.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="90d")

    if df.empty:
        raise ValueError(
            f"No data found for '{symbol}'. "
            "Check the ticker — example: TCS, RELIANCE, INFY, HDFCBANK"
        )

    return df["Close"].values.astype(float)


def predict_next_price(closes):
    """
    Predict tomorrow's closing price with NumPy linear regression.

    How np.polyfit works:
      - x = [0, 1, 2, ..., n-1]  (day index)
      - y = closing prices
      - degree=1 → fits a straight line  y = m*x + b
      - We evaluate that line at x = n  (the next day)
    """
    x = np.arange(len(closes))           # day indices
    m, b = np.polyfit(x, closes, deg=1)  # slope and intercept
    next_day_index = len(closes)          # one step ahead
    predicted = m * next_day_index + b
    return round(float(predicted), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET  → render empty form
    POST → fetch data, predict, render results
    """
    result = None
    error  = None

    if request.method == "POST":
        raw = request.form.get("symbol", "").strip()

        if not raw:
            error = "Please enter a stock symbol (e.g. TCS or RELIANCE)."
        else:
            try:
                symbol  = normalize_symbol(raw)
                closes  = fetch_closes(symbol)

                current_price   = round(float(closes[-1]), 2)
                predicted_price = predict_next_price(closes)
                change_pct      = round(
                    (predicted_price - current_price) / current_price * 100, 2
                )

                if predicted_price > current_price:
                    signal       = "BUY"
                    signal_class = "buy"
                    signal_icon  = "↑"
                else:
                    signal       = "SELL"
                    signal_class = "sell"
                    signal_icon  = "↓"

                result = {
                    "symbol":          symbol,
                    "current_price":   current_price,
                    "predicted_price": predicted_price,
                    "change_pct":      change_pct,
                    "signal":          signal,
                    "signal_class":    signal_class,
                    "signal_icon":     signal_icon,
                    "data_points":     len(closes),
                }

            except ValueError as e:
                error = str(e)
            except Exception as e:
                error = f"Unexpected error: {e}. Please try again."

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
