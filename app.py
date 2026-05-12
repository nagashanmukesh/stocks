from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# ===================== BASIC ROUTES =====================
@app.route("/")
def index():
    return send_from_directory(_HERE, "index.html")


@app.route("/style.css")
def serve_css():
    return send_from_directory(_HERE, "style.css")


@app.route("/ping")
def ping():
    return "pong ✅ Server is running!"


@app.route("/analyze", methods=["POST"])
def analyze():
    # Minimal response for testing
    return jsonify({
        "symbol": "RELIANCE",
        "company_name": "Reliance Industries Ltd",
        "current_price": 2450.75,
        "predicted_price": 2480.50,
        "recommendation": "BUY",
        "trend": "UPTREND",
        "quant_score": 68,
        "explanations": [
            "This is a basic test response.",
            "If you see this, backend is working."
        ],
        "chart_dates": ["1 May", "2 May", "3 May"],
        "chart_closes": [2400, 2420, 2450]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
