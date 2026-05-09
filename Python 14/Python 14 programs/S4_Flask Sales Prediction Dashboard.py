# ===============================================================
# SALES PREDICTION DASHBOARD (UPDATED UI)
# Flask + TensorFlow + Separate Graph Capsule + White Chart BG
# ===============================================================
print("SY-5, Kevin Victor, Roll No.-30")
from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

print("===================================================")
print("SEMICONDUCTOR CHIP SALES MODEL - TRAINING STARTED")
print("===================================================")

# ===============================================================
# 1. DATASET
# ===============================================================

np.random.seed(42)
samples = 3200

processor_gen = np.random.choice([1, 2, 3, 4], samples)

phone_type = np.random.choice(
    ["gaming", "ai_powered", "high_performance"], samples
)

market_scenario = np.random.choice(
    ["weak", "normal", "strong", "boom"], samples
)

user_interest = np.random.uniform(1, 10, samples)
services = np.random.uniform(1, 10, samples)
brand_power = np.random.uniform(1, 10, samples)
battery_eff = np.random.uniform(1, 10, samples)
price_comp = np.random.uniform(1, 10, samples)

# ===============================================================
# 2. ENCODING
# ===============================================================

phone_encoder = LabelEncoder()
market_encoder = LabelEncoder()

phone_enc = phone_encoder.fit_transform(phone_type)
market_enc = market_encoder.fit_transform(market_scenario)

# ===============================================================
# 3. FEATURE ENGINEERING
# ===============================================================

perf_score = processor_gen * 2.2 + services * 0.8 + battery_eff * 0.6
demand_score = user_interest * 1.5 + brand_power * 1.2 + price_comp * 0.9

market_boost = {
    "weak": -20,
    "normal": 0,
    "strong": 25,
    "boom": 55
}

sales = (
    processor_gen * 35 +
    user_interest * 12 +
    services * 8 +
    brand_power * 9 +
    battery_eff * 7 +
    price_comp * 8 +
    perf_score * 6 +
    demand_score * 5 +
    np.array([market_boost[x] for x in market_scenario]) +
    np.random.normal(0, 18, samples)
)

sales = np.maximum(sales, 40)

# ===============================================================
# 4. DATAFRAME
# ===============================================================

df = pd.DataFrame({
    "processor_gen": processor_gen,
    "phone_type": phone_enc,
    "market": market_enc,
    "user_interest": user_interest,
    "services": services,
    "brand_power": brand_power,
    "battery_eff": battery_eff,
    "price_comp": price_comp,
    "perf_score": perf_score,
    "demand_score": demand_score,
    "sales": sales
})

X = df.drop("sales", axis=1).values
y = df["sales"].values

# ===============================================================
# 5. SPLIT + SCALE
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================================================
# 6. MODEL
# ===============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.003),
    loss="mse",
    metrics=["mae"]
)

model.fit(
    X_train, y_train,
    epochs=28,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test, verbose=0)

print("\n===== MODEL READY =====")
print("Loss:", round(loss, 2))
print("MAE :", round(mae, 2))

# ===============================================================
# CHART FUNCTION
# ===============================================================

def create_chart(pred, vals):

    labels = [
        "Sales", "Interest", "Services",
        "Brand", "Battery", "Price"
    ]

    values = [
        pred,
        float(vals["user_interest"]) * 25,
        float(vals["services"]) * 25,
        float(vals["brand_power"]) * 25,
        float(vals["battery_eff"]) * 25,
        float(vals["price_comp"]) * 25
    ]

    plt.figure(figsize=(9, 4.8), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    bars = plt.bar(labels, values)

    colors = [
        "#00bcd4", "#2196f3", "#4caf50",
        "#ff9800", "#e91e63", "#9c27b0"
    ]

    for b, c in zip(bars, colors):
        b.set_color(c)

    plt.title("Sales vs Demand Drivers", fontsize=14, weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, facecolor="white")
    plt.close()
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")

# ===============================================================
# FRONTEND
# ===============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Chip Sales Dashboard</title>

<style>

body{
margin:0;
font-family:Arial;
background:
linear-gradient(135deg,#08111f,#111827,#0f172a);
color:white;
padding:35px;
}

.wrapper{
max-width:1100px;
margin:auto;
display:grid;
grid-template-columns:1fr 1fr;
gap:28px;
align-items:start;
}

.card, .graphcard{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(16px);
border:1px solid rgba(255,255,255,0.12);
border-radius:24px;
padding:28px;
box-shadow:0 0 30px rgba(0,255,255,0.08);
animation:rise 1s ease;
}

@keyframes rise{
from{opacity:0;transform:translateY(25px);}
to{opacity:1;transform:translateY(0);}
}

h1{
margin-top:0;
text-align:center;
font-size:28px;
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
gap:14px;
margin-top:18px;
}

button{
flex:1;
padding:14px;
border:none;
border-radius:12px;
font-size:16px;
font-weight:bold;
cursor:pointer;
transition:0.3s;
}

.predict{
background:linear-gradient(90deg,#06b6d4,#3b82f6);
color:white;
}

.reset{
background:linear-gradient(90deg,#ef4444,#f97316);
color:white;
}

button:hover{
transform:scale(1.03);
}

.result{
margin-top:18px;
padding:16px;
border-radius:16px;
background:rgba(255,255,255,0.08);
font-size:22px;
text-align:center;
}

.graphcard{
background:rgba(255,255,255,0.10);
}

.graphcard img{
width:100%;
border-radius:18px;
background:white;
padding:10px;
box-sizing:border-box;
}

.note{
text-align:center;
opacity:0.8;
margin-top:10px;
}

@media(max-width:980px){
.wrapper{
grid-template-columns:1fr;
}
}

</style>
</head>

<body>

<div class="wrapper">

<div class="card">

<h1>📈 Chip Sales Dashboard</h1>

<form method="POST">

<div class="grid">

<select name="processor_gen">
{% for g in [1,2,3,4] %}
<option value="{{g}}" {% if vals.processor_gen==g|string %}selected{% endif %}>
Gen {{g}}
</option>
{% endfor %}
</select>

<select name="phone_type">
{% for t in phone_types %}
<option value="{{t}}" {% if vals.phone_type==t %}selected{% endif %}>
{{t}}
</option>
{% endfor %}
</select>

<select name="market">
{% for m in markets %}
<option value="{{m}}" {% if vals.market==m %}selected{% endif %}>
{{m}}
</option>
{% endfor %}
</select>

<input type="number" step="0.1" min="1" max="10"
name="user_interest"
value="{{vals.user_interest}}"
placeholder="User Interest">

<input type="number" step="0.1" min="1" max="10"
name="services"
value="{{vals.services}}"
placeholder="Extra Services">

<input type="number" step="0.1" min="1" max="10"
name="brand_power"
value="{{vals.brand_power}}"
placeholder="Brand Power">

<input type="number" step="0.1" min="1" max="10"
name="battery_eff"
value="{{vals.battery_eff}}"
placeholder="Battery Efficiency">

<input type="number" step="0.1" min="1" max="10"
name="price_comp"
value="{{vals.price_comp}}"
placeholder="Price Competitiveness">

</div>

<div class="buttons">
<button class="predict" type="submit">Predict</button>
<button class="reset" type="button"
onclick="window.location='/'">Reset</button>
</div>

</form>

{% if result %}
<div class="result">{{result}}</div>
{% endif %}

<div class="note">
Predicted sales shown in thousand units
</div>

</div>

<div class="graphcard">

<h1>📊 Analytics Capsule</h1>

{% if chart %}
<img src="data:image/png;base64,{{chart}}">
{% else %}
<div class="note" style="padding:100px 20px;">
Run a prediction to generate the chart.
</div>
{% endif %}

</div>

</div>

</body>
</html>
"""

# ===============================================================
# ROUTE
# ===============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    chart = None

    vals = {
        "processor_gen": "4",
        "phone_type": "ai_powered",
        "market": "strong",
        "user_interest": "",
        "services": "",
        "brand_power": "",
        "battery_eff": "",
        "price_comp": ""
    }

    if request.method == "POST":

        vals = request.form.to_dict()

        gen = int(vals["processor_gen"])
        ptype = vals["phone_type"]
        market = vals["market"]

        ui = float(vals["user_interest"])
        srv = float(vals["services"])
        brand = float(vals["brand_power"])
        batt = float(vals["battery_eff"])
        price = float(vals["price_comp"])

        p_enc = phone_encoder.transform([ptype])[0]
        m_enc = market_encoder.transform([market])[0]

        perf = gen * 2.2 + srv * 0.8 + batt * 0.6
        demand = ui * 1.5 + brand * 1.2 + price * 0.9

        row = np.array([[
            gen, p_enc, m_enc,
            ui, srv, brand, batt, price,
            perf, demand
        ]])

        row = scaler.transform(row)

        pred = model.predict(row, verbose=0)[0][0]

        result = f"Predicted Sales: {pred:.2f} Thousand Units"

        chart = create_chart(pred, vals)

    return render_template_string(
        HTML,
        result=result,
        chart=chart,
        vals=vals,
        phone_types=["gaming", "ai_powered", "high_performance"],
        markets=["weak", "normal", "strong", "boom"]
    )

# ===============================================================
# MAIN
# ===============================================================

if __name__ == "__main__":
    app.run(debug=True)