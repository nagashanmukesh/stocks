# 📈 NSEPredict — Indian Stock Price Predictor

A beginner-friendly Flask web app that predicts NSE stock prices using Linear Regression.

---

## 🗂️ Project Structure

```
stock-app/
├── app.py               ← Flask backend + ML logic
├── requirements.txt     ← Python dependencies
├── Procfile             ← For Gunicorn (deployment)
├── templates/
│   └── index.html       ← HTML UI
└── static/
    └── style.css        ← Dark dashboard CSS
```

---

## 💻 Local Setup (Step-by-Step)

### Step 1 — Make sure Python is installed
```bash
python --version
# Should show Python 3.9 or higher
```

### Step 2 — Create a project folder and enter it
```bash
mkdir stock-app
cd stock-app
```

### Step 3 — Copy all the project files into this folder
(copy app.py, requirements.txt, Procfile, and the templates/ and static/ folders)

### Step 4 — Create a virtual environment
```bash
python -m venv venv
```

### Step 5 — Activate the virtual environment

On Windows:
```bash
venv\Scripts\activate
```

On Mac/Linux:
```bash
source venv/bin/activate
```

### Step 6 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 7 — Run the app
```bash
python app.py
```

### Step 8 — Open in browser
Visit: http://localhost:5000

---

## 🌐 Deploy to Render (Free Hosting)

### Step 1 — Push your code to GitHub
1. Create a free account at https://github.com
2. Create a new repository called `nse-predict`
3. Upload all your project files to the repo

### Step 2 — Create a free Render account
Visit: https://render.com and sign up for free

### Step 3 — Create a new Web Service on Render
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub account
3. Select your `nse-predict` repository

### Step 4 — Configure the Render settings
Fill in exactly these settings:

| Setting | Value |
|---|---|
| **Name** | nse-predict (or anything you like) |
| **Region** | Singapore (closest to India) |
| **Branch** | main |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Instance Type** | Free |

### Step 5 — Click "Create Web Service"
Render will automatically build and deploy your app.
It takes about 2–3 minutes for the first deploy.

### Step 6 — Access your live app
Render will give you a URL like:
`https://nse-predict.onrender.com`

> ⚠️ Free Render apps "sleep" after 15 minutes of no traffic.
> The first request after sleep takes ~30 seconds to load. This is normal!

---

## 🧠 How the Prediction Works

1. **Data Fetch**: We use `yfinance` to download the last 60 days of closing prices for the given NSE stock.

2. **Features (X)**: We simply use day numbers — [0, 1, 2, ..., 59].

3. **Target (y)**: The actual closing prices for each day.

4. **Model**: We fit a `LinearRegression` line (y = mx + b) through these 60 points.

5. **Predict**: We ask the model to predict for day 60 (tomorrow).

6. **Signal**:
   - If predicted price > today's price → **BUY**
   - If predicted price ≤ today's price → **SELL**

---

## ⚠️ Limitations of This Simple Model

- **Linear only**: Real stock prices are NOT linear. This model assumes a straight-line trend, which is a massive simplification.
- **No external factors**: News, earnings, RBI policy, global events — none of these are considered.
- **Only uses price history**: Volume, technical indicators, fundamentals are ignored.
- **60 days may not be enough**: Some stocks need years of data for meaningful patterns.
- **No train/test split**: We're training and predicting on the same data range (overfitting risk).

**This app is for LEARNING purposes only. Never make real investment decisions based on it.**

---

## 🚀 Future Improvements You Can Add

1. **More ML models**: Try RandomForest, LSTM (deep learning), or ARIMA for time-series.
2. **Technical indicators**: Add RSI, MACD, Bollinger Bands as features.
3. **Chart visualization**: Use Chart.js or Plotly to show price history.
4. **Confidence score**: Show how "confident" the model is.
5. **Multi-day forecast**: Predict 7 days instead of just 1.
6. **Portfolio tracker**: Let users track multiple stocks.
7. **News sentiment**: Fetch news headlines and add sentiment as a feature.
8. **Train/test split**: Properly evaluate model accuracy on unseen data.

---

*Built with ❤️ for learning · Not financial advice*
