# ===============================================================
# FULL FLASK FRONTEND + POST API + TENSORFLOW DRONE MODEL
# ===============================================================
# FEATURES:
# 1. Trains delivery-drone TensorFlow model on startup
# 2. Flask frontend (browser UI)
# 3. POST API endpoint accepting JSON (Postman / Python / JS)
# 4. Predefined test buttons for quick evaluation
# 5. Background checks:
#    - TensorFlow import check
#    - Model training status
#    - Input validation
#    - NaN / Inf check
#    - Shape check
#    - Prediction exception handling
# 6. Human-readable output + JSON API output
#
# RUN:
# python app.py
#
# OPEN:
# http://127.0.0.1:5000
#
# POSTMAN:
# POST http://127.0.0.1:5000/predict
# Body -> raw -> JSON
# ===============================================================

from flask import Flask, request, jsonify, render_template_string
import numpy as np
import traceback

print("SY-5, Kevin Victor, Roll No.-30")
print("===== DELIVERY DRONE MODEL + FLASK FRONTEND =====")

# ---------------------------------------------------------------
# BACKGROUND CHECK 1 : TensorFlow Import
# ---------------------------------------------------------------
try:
    import tensorflow as tf
    TF_OK = True
except Exception as e:
    TF_OK = False
    print("TensorFlow Import Failed:", e)

if not TF_OK:
    raise SystemExit("TensorFlow required.")

# ===============================================================
# STEP 1 : BUILD DATASET
# ===============================================================
np.random.seed(42)

samples = 300

path_eff = np.random.uniform(0.4, 1.0, samples)
stability = np.random.uniform(0.5, 1.0, samples)
obstacles = np.random.uniform(0.0, 1.0, samples)
payload = np.random.uniform(0.5, 5.0, samples)
battery = np.random.uniform(0.3, 1.0, samples)
weather = np.random.uniform(0.0, 1.0, samples)

X_random = np.column_stack([
    path_eff, stability, obstacles,
    payload, battery, weather
]).astype(np.float32)

# success-biased extra data
success_samples = 200

X_success = np.column_stack([
    np.random.uniform(0.75, 1.0, success_samples),
    np.random.uniform(0.75, 1.0, success_samples),
    np.random.uniform(0.0, 0.3, success_samples),
    np.random.uniform(0.5, 3.0, success_samples),
    np.random.uniform(0.7, 1.0, success_samples),
    np.random.uniform(0.0, 0.4, success_samples)
]).astype(np.float32)

X = np.vstack([X_random, X_success])

y = ((X[:,0] > 0.6) &
     (X[:,1] > 0.6) &
     (X[:,2] < 0.6) &
     (X[:,4] > 0.5) &
     (X[:,5] < 0.7)).astype(int)

# ===============================================================
# STEP 2 : SIMPLE SPLIT
# ===============================================================
split = int(len(X) * 0.8)

X_train = X[:split]
X_test  = X[split:]

y_train = y[:split]
y_test  = y[split:]

# ===============================================================
# STEP 3 : NORMALIZATION
# ===============================================================
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-6

X_train_n = (X_train - mean) / std
X_test_n  = (X_test  - mean) / std

# ===============================================================
# STEP 4 : FEATURE ENGINEERING
# ===============================================================
def engineer_features(data):
    pe, st, ob, pl, bt, wt = data.T

    navigation_score   = pe * bt
    risk_factor        = ob + wt
    load_efficiency    = pl / (bt + 1e-5)
    stability_margin   = st * bt
    safety_index       = (1 - ob) * (1 - wt)
    efficiency_balance = pe * st / (pl + 1e-5)

    return np.column_stack([
        navigation_score,
        risk_factor,
        load_efficiency,
        stability_margin,
        safety_index,
        efficiency_balance
    ])

X_train_f = engineer_features(X_train_n)
X_test_f  = engineer_features(X_test_n)

# ===============================================================
# STEP 5 : BUILD MODEL
# ===============================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(6,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------------------
# BACKGROUND CHECK 2 : Training
# ---------------------------------------------------------------
print("Training model...")

model.fit(
    X_train_f, y_train,
    epochs=20,
    batch_size=16,
    verbose=0
)

loss, acc = model.evaluate(X_test_f, y_test, verbose=0)
print("Model Ready | Test Accuracy:", round(float(acc), 4))

# ===============================================================
# STEP 6 : FLASK APP
# ===============================================================
app = Flask(__name__)

# ===============================================================
# HELPERS
# ===============================================================
def validate_input(arr):
    if len(arr) != 6:
        return False, "Need exactly 6 values."

    arr = np.array(arr, dtype=np.float32)

    if np.isnan(arr).any():
        return False, "NaN detected."

    if np.isinf(arr).any():
        return False, "Infinity detected."

    return True, arr


def predict_sample(sample):

    sample = np.array(sample, dtype=np.float32).reshape(1, 6)

    # preprocess
    sample_n = (sample - mean) / std
    sample_f = engineer_features(sample_n)

    pred = model.predict(sample_f, verbose=0)[0]

    cls = int(np.argmax(pred))
    prob_success = float(pred[1])
    prob_fail = float(pred[0])

    decision = "TRIP SUCCESS" if cls == 1 else "TRIP FAILURE"

    return {
        "input": sample.flatten().tolist(),
        "prediction_class": cls,
        "decision": decision,
        "success_probability": round(prob_success, 4),
        "failure_probability": round(prob_fail, 4)
    }

# ===============================================================
# FRONTEND HTML
# ===============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Drone Predictor</title>
<style>
body { font-family: Arial; margin:40px; background:#f5f5f5; }
.card { background:white; padding:20px; border-radius:12px; max-width:800px; }
button { padding:10px; margin:4px; cursor:pointer; }
pre { background:#111; color:#0f0; padding:15px; }
</style>
</head>
<body>
<div class="card">
<h2>Delivery Drone Predictor</h2>
<p>Click predefined test conditions:</p>

<button onclick="runTest([0.8,0.9,0.2,2.0,0.9,0.3])">Test 1</button>
<button onclick="runTest([0.5,0.6,0.7,3.0,0.6,0.8])">Test 2</button>
<button onclick="runTest([0.7,0.8,0.4,4.5,0.7,0.5])">Test 3</button>
<button onclick="runTest([0.4,0.5,0.8,3.5,0.4,0.9])">Test 4</button>
<button onclick="runTest([0.9,0.95,0.1,1.5,0.95,0.2])">Test 5</button>

<p>Manual JSON API also available:</p>
<p><b>POST /predict</b></p>

<pre id="output">Awaiting input...</pre>
</div>

<script>
async function runTest(arr){
    const res = await fetch('/predict', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({features: arr})
    });

    const data = await res.json();
    document.getElementById("output").textContent =
        JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""

# ===============================================================
# ROUTES
# ===============================================================
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


# ---------------------------------------------------------------
# POST API ENDPOINT
# Accepts JSON:
# {
#   "features":[0.8,0.9,0.2,2.0,0.9,0.3]
# }
# ---------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        features = data.get("features", None)

        if features is None:
            return jsonify({"error": "Missing 'features' key"}), 400

        ok, validated = validate_input(features)

        if not ok:
            return jsonify({"error": validated}), 400

        result = predict_sample(validated)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# ===============================================================
# HEALTH ROUTE
# ===============================================================
@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "tensorflow": True,
        "model_ready": True
    })


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    print("Flask Server Starting...")
    print("Open: http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)