# ==========================================================
# HOUSE PRICE PREDICTION API + WEB APP
# TEST DEPLOYED APIs USING:
# 1. Browser
# 2. Postman
# 3. Curl
#
# BASED ON YOUR ORIGINAL HOUSE PRICE MODEL
#
# FEATURES:
# ✔ TensorFlow model training
# ✔ Flask frontend UI
# ✔ REST API endpoint
# ✔ JSON responses
# ✔ Browser test route
# ✔ Logging
# ✔ Health checks
#
# RUN:
# python app.py
#
# THEN TEST:
# Browser UI:
# http://127.0.0.1:5000/
#
# API Health:
# http://127.0.0.1:5000/api/health
#
# Browser GET Test:
# http://127.0.0.1:5000/api/predict?area=1500&bedrooms=3&locality=Baner&amenities=7&club=1&city_dist=6&station_dist=4&airport_dist=11
#
# Postman:
# POST http://127.0.0.1:5000/api/predict
# Body -> raw -> JSON
# ==========================================================

print("SY-5, Kevin Victor, Roll No.-30")

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import traceback
print("SY-5, Kevin Victor, Roll No.-30")
# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# APP
# ==========================================================

app = Flask(__name__)

# ==========================================================
# MODEL TRAINING
# ==========================================================

print("===== TRAINING HOUSE PRICE MODEL =====")

np.random.seed(42)
samples = 2500

localities = [
    "Baner", "Kothrud", "Wakad",
    "Hinjewadi", "Viman Nagar",
    "Aundh", "Pimple Saudagar"
]

area = np.random.randint(600, 4000, samples)
bedrooms = np.random.choice([2, 2.5, 3, 3.5, 4, 4.5, 5], samples)
locality = np.random.choice(localities, samples)

amenities = np.random.randint(1, 11, samples)
club = np.random.randint(0, 2, samples)
city_dist = np.random.uniform(1, 25, samples)
station_dist = np.random.uniform(1, 20, samples)
airport_dist = np.random.uniform(5, 35, samples)

loc_boost = {
    "Baner": 35,
    "Kothrud": 32,
    "Wakad": 26,
    "Hinjewadi": 24,
    "Viman Nagar": 34,
    "Aundh": 38,
    "Pimple Saudagar": 27
}

price = (
    area * 0.012 +
    bedrooms * 18 +
    amenities * 3.5 +
    club * 10 +
    np.array([loc_boost[x] for x in locality]) -
    city_dist * 1.2 -
    station_dist * 0.5 -
    airport_dist * 0.3 +
    np.random.normal(0, 8, samples)
)

price = np.maximum(price, 35)

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "locality": locality,
    "amenities": amenities,
    "club": club,
    "city_dist": city_dist,
    "station_dist": station_dist,
    "airport_dist": airport_dist,
    "price": price
})

# Encode locality
le = LabelEncoder()
df["locality_enc"] = le.fit_transform(df["locality"])

X = df[
    [
        "area", "bedrooms", "locality_enc",
        "amenities", "club",
        "city_dist", "station_dist",
        "airport_dist"
    ]
].values

y = df["price"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(8,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss="mse",
    metrics=["mae"]
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test, verbose=0)

print("MODEL READY")
print("MAE:", round(mae, 2), "Lakhs")

# ==========================================================
# HELPER FUNCTION
# ==========================================================

def predict_price(data):

    loc = data["locality"]

    loc_enc = le.transform([loc])[0]

    arr = np.array([[
        float(data["area"]),
        float(data["bedrooms"]),
        float(loc_enc),
        float(data["amenities"]),
        float(data["club"]),
        float(data["city_dist"]),
        float(data["station_dist"]),
        float(data["airport_dist"])
    ]])

    arr = scaler.transform(arr)

    pred = model.predict(arr, verbose=0)[0][0]

    if pred >= 100:
        display = f"₹ {pred/100:.2f} Crores"
    else:
        display = f"₹ {pred:.2f} Lakhs"

    return float(pred), display


# ==========================================================
# FRONTEND UI
# ==========================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>House Price Predictor</title>
<style>
body{
font-family:Arial;
background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
color:white;
display:flex;
justify-content:center;
align-items:center;
min-height:100vh;
margin:0;
}
.box{
width:700px;
padding:30px;
border-radius:20px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(15px);
}
.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:12px;
}
input,select{
padding:12px;
border:none;
border-radius:10px;
}
button{
padding:12px;
margin-top:15px;
width:100%;
border:none;
border-radius:10px;
font-weight:bold;
cursor:pointer;
background:#06b6d4;
color:white;
}
.result{
margin-top:20px;
padding:15px;
background:rgba(255,255,255,0.08);
border-radius:10px;
font-size:22px;
text-align:center;
}
</style>
</head>
<body>
<div class="box">
<h1>🏠 House Price Predictor</h1>
<form method="POST">
<div class="grid">
<input name="area" placeholder="Area">
<input name="bedrooms" placeholder="Bedrooms">
<select name="locality">
{% for x in localities %}
<option value="{{x}}">{{x}}</option>
{% endfor %}
</select>
<input name="amenities" placeholder="Amenities">
<input name="club" placeholder="Club 0/1">
<input name="city_dist" placeholder="City Dist">
<input name="station_dist" placeholder="Station Dist">
<input name="airport_dist" placeholder="Airport Dist">
</div>
<button type="submit">Predict</button>
</form>

{% if result %}
<div class="result">{{result}}</div>
{% endif %}
</div>
</body>
</html>
"""

# ==========================================================
# WEB ROUTE
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":
        try:
            val = request.form.to_dict()
            _, result = predict_price(val)

        except Exception as e:
            result = "Invalid Input"

    return render_template_string(
        HTML,
        result=result,
        localities=localities
    )

# ==========================================================
# API HEALTH CHECK
# ==========================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model": "loaded",
        "service": "House Price Predictor API"
    })

# ==========================================================
# API PREDICT (GET + POST)
# ==========================================================

@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():

    try:

        if request.method == "POST":
            data = request.get_json()

        else:
            data = request.args.to_dict()

        pred, display = predict_price(data)

        logger.info("Prediction successful")

        return jsonify({
            "success": True,
            "predicted_price_raw_lakhs": round(pred, 2),
            "display_price": display,
            "inputs": data
        })

    except Exception as e:

        logger.error(traceback.format_exc())

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\n===== SERVER STARTED =====")
    print("Browser UI:")
    print("http://127.0.0.1:5000/\n")

    print("API Health:")
    print("http://127.0.0.1:5000/api/health\n")

    print("GET API Test:")
    print("http://127.0.0.1:5000/api/predict?area=1500&bedrooms=3&locality=Baner&amenities=7&club=1&city_dist=6&station_dist=4&airport_dist=11\n")

    app.run(debug=True)