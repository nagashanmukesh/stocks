import os
from flask import Flask, send_from_directory, jsonify

app = Flask(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# Serve Frontend
@app.route("/")
def index():
    return send_from_directory(_HERE, "index.html")

@app.route("/style.css")
def css():
    return send_from_directory(_HERE, "style.css")

@app.route("/ping")
def ping():
    return "pong"

# Simple Analyze for testing
@app.route("/analyze", methods=["POST"])
def analyze():
    return jsonify({
        "symbol": "TEST",
        "company_name": "Test Company",
        "current_price": 1000,
        "predicted_price": 1025,
        "recommendation": "BUY",
        "trend": "UPTREND",
        "explanations": ["This is a test response"],
        "chart_dates": ["Day1", "Day2"],
        "chart_closes": [980, 1000]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
