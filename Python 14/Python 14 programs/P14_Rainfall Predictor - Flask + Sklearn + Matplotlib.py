# ============================================================
# RAINFALL PREDICTOR (FLASK + SKLEARN ONLY)
# FIXED VERSION (NO TF, NO CRASH, WORKING UI)
# ============================================================

print("SY-5, Kevin Victor, Roll No.-30")

from flask import Flask, request, jsonify, render_template_string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ✅ CRITICAL FIX (prevents white screen crash)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# ============================================================
# FEATURES
# ============================================================

FEATURE_NAMES = [
    "Temperature (°C)", "Humidity (%)", "Pressure (hPa)",
    "Moisture Inflow", "Vertical Mixing", "Jet Stream",
    "Ocean Current", "ENSO Index", "Vorticity", "Wind Shear"
]

# ============================================================
# DATASET
# ============================================================

np.random.seed(42)
n = 600

temp = np.random.uniform(20, 40, n)
humidity = np.random.uniform(40, 100, n)
pressure = np.random.uniform(990, 1020, n)
moisture = np.random.uniform(0, 10, n)
vmix = np.random.uniform(0, 10, n)
jet = np.random.uniform(0, 10, n)
ocean = np.random.uniform(0, 10, n)
enso = np.random.uniform(-2, 2, n)
vorticity = np.random.uniform(0, 10, n)
wind_shear = np.random.uniform(0, 10, n)

X = np.column_stack([
    temp, humidity, pressure, moisture,
    vmix, jet, ocean, enso, vorticity, wind_shear
])

# Soft labels
rain_prob = (
    0.25*(temp/40) +
    0.3*(humidity/100) +
    0.2*(moisture/10) +
    0.1*(vmix/10) +
    0.1*((jet+vorticity+wind_shear)/30) -
    0.15*((pressure-990)/30)
)

rain_prob = np.clip(rain_prob, 0, 1)

# ============================================================
# SKLEARN PIPELINE
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, rain_prob, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully")

# ============================================================
# TEST SAMPLES
# ============================================================

TEST_SAMPLES = [
    [32, 85, 1005, 8, 7, 6, 7, 0.5, 8, 7],
    [28, 60, 1015, 3, 4, 2, 3, -1.2, 2, 3],
    [35, 75, 1008, 7, 6, 8, 6, 1.2, 7, 6],
    [30, 90, 1002, 9, 8, 7, 8, 0.8, 9, 8],
    [25, 50, 1018, 2, 3, 1, 2, -1.5, 1, 2]
]

# ============================================================
# CONTRIBUTION PLOT (COEFFICIENT BASED)
# ============================================================

def generate_plot(sample_scaled):

    coeffs = model.coef_
    contrib = sample_scaled.flatten() * coeffs

    plt.figure(figsize=(8,4))
    plt.barh(FEATURE_NAMES, contrib)
    plt.title("Feature Contribution (Linear Model)")
    plt.grid(True, linestyle="--", alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()

    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ============================================================
# API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        idx = int(request.json["sample_index"])
    except:
        return jsonify({"error": "Invalid input"}), 400

    if idx < 0 or idx >= len(TEST_SAMPLES):
        return jsonify({"error": "Index out of range"}), 400

    sample = np.array(TEST_SAMPLES[idx]).reshape(1,-1)
    scaled = scaler.transform(sample)

    prob = model.predict(scaled)[0]
    prob = float(np.clip(prob, 0, 1))

    pred = 1 if prob >= 0.5 else 0

    plot = generate_plot(scaled)

    return jsonify({
        "features": FEATURE_NAMES,
        "values": TEST_SAMPLES[idx],
        "probability": round(prob, 4),
        "prediction": pred,
        "plot": plot
    })

# ============================================================
# FRONTEND (FIXED — GUARANTEED RENDER)
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Rainfall Predictor</title>

<style>
body{
margin:0;
font-family:Arial;
background:#0f172a;
color:white;
text-align:center;
}

.box{
margin:40px auto;
width:600px;
padding:20px;
background:#1e293b;
border-radius:12px;
}

button{
margin:5px;
padding:10px;
cursor:pointer;
}

img{
width:100%;
margin-top:15px;
}
</style>

<script>
function predict(i){
fetch("/predict",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({sample_index:i})
})
.then(r=>r.json())
.then(data=>{

let table="<table border='1' style='margin:auto'>";
for(let j=0;j<data.features.length;j++){
table+="<tr><td>"+data.features[j]+"</td><td>"+data.values[j]+"</td></tr>";
}
table+="</table>";

document.getElementById("data").innerHTML=table;

document.getElementById("result").innerHTML=
"Probability: "+data.probability+" | Prediction: "+data.prediction;

document.getElementById("plot").src="data:image/png;base64,"+data.plot;

});
}
</script>

</head>

<body>

<div class="box">
<h2>Rainfall Predictor</h2>

<button onclick="predict(0)">Sample 1</button>
<button onclick="predict(1)">Sample 2</button>
<button onclick="predict(2)">Sample 3</button>
<button onclick="predict(3)">Sample 4</button>
<button onclick="predict(4)">Sample 5</button>

<div id="data"></div>
<div id="result"></div>
<img id="plot"/>

</div>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)