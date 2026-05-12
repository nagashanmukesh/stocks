from flask import Flask, render_template, request
import yfinance as yf
import numpy as np

app = Flask(
    __name__,
    template_folder='.',
    static_folder='.'
)

# Serve CSS manually
@app.route('/style.css')
def serve_css():
    return app.send_static_file('style.css')


def normalize_symbol(symbol):

    symbol = symbol.strip().upper()

    if not symbol.endswith(".NS"):
        symbol += ".NS"

    return symbol


def fetch_stock_data(symbol):

    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period="3mo",
        interval="1d",
        auto_adjust=True
    )

    if df.empty:
        raise ValueError("Invalid stock symbol.")

    return df


def predict_next_price(closes):

    x = np.arange(len(closes))

    slope, intercept = np.polyfit(x, closes, 1)

    next_day = len(closes)

    predicted = slope * next_day + intercept

    return round(float(predicted), 2)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        try:

            symbol = normalize_symbol(
                request.form.get("symbol", "")
            )

            df = fetch_stock_data(symbol)

            closes = df["Close"].values.astype(float)

            current_price = round(float(closes[-1]), 2)

            predicted_price = predict_next_price(closes)

            change_pct = round(
                ((predicted_price - current_price) / current_price) * 100,
                2
            )

            if predicted_price > current_price:

                signal = "BUY"
                signal_class = "buy"
                signal_icon = "↑"

            else:

                signal = "SELL"
                signal_class = "sell"
                signal_icon = "↓"

            result = {
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "change_pct": change_pct,
                "signal": signal,
                "signal_class": signal_class,
                "signal_icon": signal_icon
            }

        except Exception as e:

            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
