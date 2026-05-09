# ==========================================================
# HOUSE PRICE PREDICTION WEB APP (FLASK + TENSORFLOW)
# FULL CORRECTED VERSION
# CHANGES MADE:
# 1. Input fields RETAIN values after Predict
# 2. Separate Reset button clears all fields
# 3. Beautiful responsive UI
# 4. TensorFlow model trains in terminal only
# ==========================================================
print("SY-5, Kevin Victor, Roll No.-30")
from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

print("===== HOUSE PRICE MODEL TRAINING STARTED =====")

# ==========================================================
# 1. SYNTHETIC DATASET (INDIAN MARKET STYLE)
# ==========================================================

np.random.seed(42)
samples = 2500

localities = [
    "Baner", "Kothrud", "Wakad", "Hinjewadi",
    "Viman Nagar", "Aundh", "Pimple Saudagar"
]

area = np.random.randint(600, 4000, samples)
bedrooms = np.random.choice(
    [2, 2.5, 3, 3.5, 4, 4.5, 5], samples
)

locality = np.random.choice(localities, samples)

amenities = np.random.randint(1, 11, samples)
club = np.random.randint(0, 2, samples)
city_dist = np.random.uniform(1, 25, samples)
station_dist = np.random.uniform(1, 20, samples)
airport_dist = np.random.uniform(5, 35, samples)

# ==========================================================
# 2. TARGET PRICE (Lakhs)
# ==========================================================

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

# ==========================================================
# 3. DATAFRAME
# ==========================================================

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

# ==========================================================
# 4. ENCODE LOCALITY
# ==========================================================

le = LabelEncoder()
df["locality_enc"] = le.fit_transform(df["locality"])

X = df[
    [
        "area", "bedrooms", "locality_enc", "amenities",
        "club", "city_dist", "station_dist", "airport_dist"
    ]
].values

y = df["price"].values

# ==========================================================
# 5. SPLIT + SCALE
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# 6. MODEL BUILDING
# ==========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(8,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

# ==========================================================
# 7. COMPILE
# ==========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss="mse",
    metrics=["mae"]
)

# ==========================================================
# 8. TRAINING
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=35,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================================================
# 9. EVALUATION
# ==========================================================

loss, mae = model.evaluate(X_test, y_test, verbose=0)

print("\n===== MODEL READY =====")
print("Test MAE:", round(mae, 2), "Lakhs")

# ==========================================================
# FRONTEND HTML
# ==========================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>House Price Predictor</title>

<style>
body{
margin:0;
font-family:Arial;
background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
min-height:100vh;
display:flex;
justify-content:center;
align-items:center;
color:white;
}

.box{
width:700px;
padding:35px;
border-radius:22px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(16px);
box-shadow:0 0 30px rgba(0,255,255,0.18);
animation:fade 1.2s ease;
}

@keyframes fade{
from{opacity:0;transform:translateY(40px);}
to{opacity:1;transform:translateY(0);}
}

h1{
text-align:center;
margin-bottom:22px;
}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:14px;
}

input,select{
padding:12px;
border:none;
border-radius:12px;
font-size:15px;
width:100%;
box-sizing:border-box;
}

.buttons{
display:flex;
gap:15px;
margin-top:22px;
}

button{
flex:1;
padding:14px;
border:none;
border-radius:12px;
font-size:16px;
cursor:pointer;
font-weight:bold;
transition:0.3s;
}

.predict{
background:linear-gradient(90deg,#06b6d4,#3b82f6);
color:white;
}

.predict:hover{
transform:scale(1.03);
}

.reset{
background:#ef4444;
color:white;
}

.reset:hover{
transform:scale(1.03);
}

.result{
margin-top:25px;
padding:18px;
border-radius:14px;
background:rgba(255,255,255,0.08);
font-size:22px;
text-align:center;
}

small{
opacity:0.8;
display:block;
margin-top:10px;
text-align:center;
}
</style>
</head>

<body>

<div class="box">
<h1>🏠 House Price Predictor</h1>

<form method="POST">

<div class="grid">

<input type="number" step="0.01" name="area"
placeholder="Area (sq ft)"
value="{{vals.area}}"
required>

<select name="bedrooms">
{% for b in [2,2.5,3,3.5,4,4.5,5] %}
<option value="{{b}}" {% if vals.bedrooms==b|string %}selected{% endif %}>{{b}} BHK</option>
{% endfor %}
</select>

<select name="locality">
{% for loc in localities %}
<option value="{{loc}}" {% if vals.locality==loc %}selected{% endif %}>{{loc}}</option>
{% endfor %}
</select>

<input type="number" name="amenities"
value="{{vals.amenities}}"
placeholder="Amenities Count"
required>

<select name="club">
<option value="1" {% if vals.club=="1" %}selected{% endif %}>Club Membership</option>
<option value="0" {% if vals.club=="0" %}selected{% endif %}>No Club</option>
</select>

<input type="number" step="0.01" name="city_dist"
value="{{vals.city_dist}}"
placeholder="Distance to City Center (km)"
required>

<input type="number" step="0.01" name="station_dist"
value="{{vals.station_dist}}"
placeholder="Distance to Station (km)"
required>

<input type="number" step="0.01" name="airport_dist"
value="{{vals.airport_dist}}"
placeholder="Distance to Airport (km)"
required>

</div>

<div class="buttons">
<button class="predict" type="submit">Predict</button>
<button class="reset" type="button"
onclick="window.location='/'">Reset</button>
</div>

</form>

{% if result %}
<div class="result">
{{result}}
</div>
{% endif %}

<small>Indian market estimate in Lakhs / Crores</small>

</div>
</body>
</html>
"""

# ==========================================================
# ROUTE
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    vals = {
        "area": "",
        "bedrooms": "3",
        "locality": "Baner",
        "amenities": "",
        "club": "1",
        "city_dist": "",
        "station_dist": "",
        "airport_dist": ""
    }

    if request.method == "POST":

        vals = request.form.to_dict()

        area = float(vals["area"])
        bedrooms = float(vals["bedrooms"])
        locality = vals["locality"]
        amenities = float(vals["amenities"])
        club = float(vals["club"])
        city_dist = float(vals["city_dist"])
        station_dist = float(vals["station_dist"])
        airport_dist = float(vals["airport_dist"])

        loc_enc = le.transform([locality])[0]

        features = np.array([[
            area, bedrooms, loc_enc, amenities,
            club, city_dist, station_dist, airport_dist
        ]])

        features = scaler.transform(features)

        pred = model.predict(features, verbose=0)[0][0]

        if pred >= 100:
            result = f"₹ {pred/100:.2f} Crores"
        else:
            result = f"₹ {pred:.2f} Lakhs"

    return render_template_string(
        HTML,
        result=result,
        vals=vals,
        localities=localities
    )

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)