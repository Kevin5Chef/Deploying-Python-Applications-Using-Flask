# ============================================================
# DRONE SAFETY PREDICTOR API (UPDATED WITH TRANSPARENCY)
# ============================================================

print("SY-5, Kevin Victor, Roll No.-30")

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "drone_model.pkl"
SCALER_PATH = "scaler.pkl"

# ============================================================
# FEATURE NAMES (NEW)
# ============================================================

FEATURE_NAMES = [
    "Wind Speed (km/h)",
    "Visibility (0-1)",
    "Battery Level (0-1)",
    "Payload (kg)",
    "Temperature (°C)",
    "Humidity (%)"
]

# ============================================================
# DATASET + MODEL (UNCHANGED)
# ============================================================

np.random.seed(42)
samples = 500

wind = np.random.uniform(0, 20, samples)
visibility = np.random.uniform(0.3, 1.0, samples)
battery = np.random.uniform(0.2, 1.0, samples)
payload = np.random.uniform(0.5, 5.0, samples)
temperature = np.random.uniform(10, 40, samples)
humidity = np.random.uniform(30, 100, samples)

X = np.column_stack([
    wind, visibility, battery,
    payload, temperature, humidity
])

y = (
    (wind < 12) &
    (visibility > 0.5) &
    (battery > 0.5) &
    (humidity < 85)
).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

# Save
class ModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(ModelWrapper(model), f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Load
loaded_model = pickle.load(open(MODEL_PATH, "rb"))
loaded_scaler = pickle.load(open(SCALER_PATH, "rb"))

# ============================================================
# TEST SAMPLES
# ============================================================

TEST_SAMPLES = [
    [8, 0.8, 0.9, 2.0, 30, 60],
    [15, 0.4, 0.6, 3.0, 35, 80],
    [5, 0.9, 0.4, 1.5, 28, 70],
    [10, 0.7, 0.8, 4.5, 33, 90],
    [6, 0.85, 0.95, 1.0, 27, 55]
]

# ============================================================
# REASON GENERATOR
# ============================================================

def generate_reason(sample, pred):
    wind, vis, bat, pay, temp, hum = sample
    reasons = []

    if wind > 12:
        reasons.append("High wind instability")
    if vis < 0.5:
        reasons.append("Low visibility risk")
    if bat < 0.5:
        reasons.append("Insufficient battery")
    if hum > 85:
        reasons.append("High humidity interference")

    if not reasons:
        reasons.append("Optimal flying conditions")

    return "SAFE" if pred == 1 else "UNSAFE", reasons

# ============================================================
# API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    idx = int(data["sample_index"])

    sample = np.array(TEST_SAMPLES[idx]).reshape(1, -1)
    scaled = loaded_scaler.transform(sample)

    prob = loaded_model.predict(scaled)[0][0]
    pred = 1 if prob >= 0.5 else 0

    status, reasons = generate_reason(TEST_SAMPLES[idx], pred)

    return jsonify({
        "features": FEATURE_NAMES,
        "values": TEST_SAMPLES[idx],
        "probability": float(round(prob, 4)),
        "prediction": status,
        "reasons": reasons
    })

# ============================================================
# FRONTEND (UPDATED)
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Drone Safety AI</title>

<style>
body{
background:linear-gradient(135deg,#0f172a,#1e293b);
color:white;
font-family:Arial;
display:flex;
justify-content:center;
align-items:center;
height:100vh;
}

.box{
padding:30px;
border-radius:20px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(15px);
width:550px;
}

button{
margin:8px;
padding:10px;
border:none;
border-radius:10px;
cursor:pointer;
}

.reset{background:#ef4444;color:white;}

.table{
margin-top:15px;
border-collapse:collapse;
width:100%;
}

.table td{
padding:6px;
border-bottom:1px solid rgba(255,255,255,0.2);
}

.result{
margin-top:15px;
padding:12px;
background:rgba(255,255,255,0.1);
border-radius:10px;
}
</style>

<script>
function predict(i){
fetch("/predict", {
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({sample_index:i})
})
.then(res=>res.json())
.then(data=>{

// Build table
let table = "<table class='table'>";
for(let j=0;j<data.features.length;j++){
table += "<tr><td>"+data.features[j]+"</td><td>"+data.values[j]+"</td></tr>";
}
table += "</table>";

document.getElementById("data").innerHTML = table;

document.getElementById("result").innerHTML =
"Prediction: "+data.prediction+
"<br>Probability: "+data.probability+
"<br>Reasons: "+data.reasons.join(", ");
});
}

function resetUI(){
document.getElementById("data").innerHTML="";
document.getElementById("result").innerHTML="";
}
</script>

</head>

<body>

<div class="box">
<h2>🚁 Drone Safety Predictor</h2>

<button onclick="predict(0)">Sample 1</button>
<button onclick="predict(1)">Sample 2</button>
<button onclick="predict(2)">Sample 3</button>
<button onclick="predict(3)">Sample 4</button>
<button onclick="predict(4)">Sample 5</button>

<br>
<button class="reset" onclick="resetUI()">Reset</button>

<div id="data"></div>
<div id="result" class="result"></div>

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