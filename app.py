# app.py - Indian NSE Stock Price Predictor
# Uses yfinance to fetch stock data and scikit-learn for prediction

from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

# Create the Flask app
app = Flask(
    __name__,
    template_folder='.',
    static_folder='.'
)

def get_stock_data(symbol):
    """
    Fetch historical stock data from Yahoo Finance.
    Returns a DataFrame with stock info or raises an error.
    """
    # Automatically add .NS suffix if user didn't include it
    if not symbol.upper().endswith(".NS"):
        symbol = symbol.upper() + ".NS"
    else:
        symbol = symbol.upper()

    # Download last 60 days of data (enough for our simple model)
    stock = yf.Ticker(symbol)
    df = stock.history(period="60d")

    # If no data returned, the symbol is invalid
    if df.empty:
        raise ValueError(f"No data found for symbol '{symbol}'. Please check the ticker.")

    return df, symbol


def predict_next_price(df):
    """
    Predict the next day's closing price using Linear Regression.

    How it works:
    - X = day numbers (0, 1, 2, ... n)
    - y = closing prices
    - We fit a straight line through these points
    - Then predict the next point (day n+1)
    """
    # Use only the closing prices
    closes = df["Close"].values

    # Create day indices: [0, 1, 2, ..., n-1]
    X = np.arange(len(closes)).reshape(-1, 1)
    y = closes

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for the next day (day n)
    next_day = np.array([[len(closes)]])
    predicted_price = model.predict(next_day)[0]

    return round(float(predicted_price), 2)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route:
    - GET  → show the empty form
    - POST → process the stock symbol and show results
    """
    result = None
    error = None

    if request.method == "POST":
        # Get the stock symbol entered by the user
        raw_symbol = request.form.get("symbol", "").strip()

        if not raw_symbol:
            error = "Please enter a stock symbol (e.g. TCS or RELIANCE)."
        else:
            try:
                # Fetch stock data
                df, symbol = get_stock_data(raw_symbol)

                # Get the latest closing price (most recent trading day)
                current_price = round(float(df["Close"].iloc[-1]), 2)

                # Predict next day price
                predicted_price = predict_next_price(df)

                # Simple BUY / SELL signal
                if predicted_price > current_price:
                    signal = "BUY"
                    signal_class = "buy"
                    signal_icon = "↑"
                else:
                    signal = "SELL"
                    signal_class = "sell"
                    signal_icon = "↓"

                # Calculate percentage change
                change_pct = round(((predicted_price - current_price) / current_price) * 100, 2)

                result = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "signal": signal,
                    "signal_class": signal_class,
                    "signal_icon": signal_icon,
                    "change_pct": change_pct,
                    "data_points": len(df),
                }

            except ValueError as e:
                # Invalid symbol or no data
                error = str(e)
            except Exception as e:
                # Any other unexpected error
                error = f"Something went wrong: {str(e)}. Please try again."

    return render_template("index.html", result=result, error=error)


# Run the app locally on port 5000
if __name__ == "__main__":
    app.run()
