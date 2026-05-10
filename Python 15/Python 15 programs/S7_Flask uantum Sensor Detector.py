# ===============================================================
# QUANTUM SENSOR DETECTOR WEB APP
# FULL CORRECTED VERSION
# MAJOR CHANGE APPLIED:
# ===============================================================

print("SY-5, Kevin Victor, Roll No.-30")

from flask import Flask, request, render_template_string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# tensorflow
import tensorflow as tf

app = Flask(__name__)

# ===============================================================
# 1. SYNTHETIC DATASET
# ===============================================================

np.random.seed(42)
samples = 3000

em_strength    = np.random.uniform(10, 100, samples)
cosmic_flux    = np.random.uniform(5, 90, samples)
quantum_noise  = np.random.uniform(0, 10, samples)
pulse_shift    = np.random.uniform(0, 40, samples)
wave_curvature = np.random.uniform(0, 25, samples)
dark_signal    = np.random.uniform(0, 50, samples)
polarization   = np.random.uniform(0, 1, samples)
anomaly_index  = np.random.uniform(0, 15, samples)

entropy = (
    0.45 * cosmic_flux +
    0.60 * quantum_noise +
    0.50 * anomaly_index +
    0.30 * dark_signal +
    np.random.normal(0, 2, samples)
)

distance = (
    8 * pulse_shift +
    3 * wave_curvature +
    2 * em_strength -
    4 * polarization +
    np.random.normal(0, 10, samples)
)

X = np.column_stack([
    em_strength,
    cosmic_flux,
    quantum_noise,
    pulse_shift,
    wave_curvature,
    dark_signal,
    polarization,
    anomaly_index
])

Y = np.column_stack([entropy, distance])

# ===============================================================
# 2. TRAIN TEST SPLIT
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ===============================================================
# 3. MODEL A : SCIKIT-LEARN
# ===============================================================

print("Training Scikit-Learn Model...")

sk_model = MultiOutputRegressor(Ridge(alpha=3.0))
sk_model.fit(X_train_s, y_train)

pred_sk = sk_model.predict(X_test_s)
mae_sk = mean_absolute_error(y_test, pred_sk)

print("Scikit-Learn MAE:", round(mae_sk, 3))

# ===============================================================
# 4. MODEL B : TENSORFLOW
# ===============================================================

print("\nTraining TensorFlow Model...")

tf_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(2)
])

tf_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.003),
    loss="mse",
    metrics=["mae"]
)

tf_model.fit(
    X_train_s,
    y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss, mae_tf = tf_model.evaluate(X_test_s, y_test, verbose=0)

print("TensorFlow MAE:", round(mae_tf, 3))
print("\n===== MODELS READY =====")

# ===============================================================
# 5. 5 TEST SAMPLES ONLY
# ===============================================================

test_samples = [
    [80, 70, 2, 22, 8, 20, 0.90, 2],
    [35, 25, 8, 10, 5, 12, 0.40, 9],
    [95, 88, 1, 35, 18, 42, 0.95, 1],
    [55, 60, 6, 17, 9, 24, 0.65, 7],
    [70, 78, 3, 28, 14, 31, 0.82, 3]
]

# ===============================================================
# 6. GRAPH
# ===============================================================

def build_chart(entropy_val, distance_val):

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["Entropy", "Distance"]
    vals = [entropy_val, distance_val]

    bars = ax.bar(labels, vals)

    for b in bars:
        b.set_alpha(0.85)

    ax.set_title("Prediction Metrics")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Magnitude")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", facecolor="white")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode()

# ===============================================================
# 7. FRONTEND
# ===============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Quantum Sensor Detector</title>

<style>
body{
margin:0;
font-family:Arial;
background:
radial-gradient(circle at top left,#00d4ff33,transparent 30%),
radial-gradient(circle at bottom right,#8b5cf633,transparent 30%),
linear-gradient(135deg,#050816,#0f172a,#020617);
color:white;
min-height:100vh;
padding:30px;
}

.container{
max-width:1000px;
margin:auto;
padding:30px;
border-radius:24px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px);
box-shadow:0 0 30px rgba(0,255,255,0.15);
animation:rise 1s ease;
}

@keyframes rise{
from{opacity:0;transform:translateY(40px);}
to{opacity:1;transform:translateY(0);}
}

h1{text-align:center;}

select{
width:100%;
padding:14px;
margin-top:14px;
border:none;
border-radius:12px;
font-size:15px;
}

.buttons{
display:flex;
gap:14px;
margin-top:20px;
}

button{
flex:1;
padding:14px;
border:none;
border-radius:12px;
font-size:16px;
font-weight:bold;
cursor:pointer;
}

.predict{
background:linear-gradient(90deg,#06b6d4,#2563eb);
color:white;
}

.reset{
background:#ef4444;
color:white;
}

.result{
margin-top:25px;
padding:20px;
border-radius:18px;
background:rgba(255,255,255,0.06);
font-size:20px;
}

.chartbox{
margin-top:25px;
background:white;
padding:20px;
border-radius:20px;
}

.samplebox{
margin-top:20px;
padding:16px;
border-radius:16px;
background:rgba(255,255,255,0.05);
line-height:1.7;
}
</style>
</head>

<body>

<div class="container">

<h1>🛰 Quantum Sensor Detector</h1>

<form method="POST">

<select name="sample_id" required>
<option value="">Select Test Sample</option>
{% for i in range(5) %}
<option value="{{i}}" {% if vals.sample_id==i|string %}selected{% endif %}>
Sample {{i+1}}
</option>
{% endfor %}
</select>

<select name="model_choice">
<option value="sk" {% if vals.model_choice=="sk" %}selected{% endif %}>
Scikit-Learn
</option>

<option value="tf" {% if vals.model_choice=="tf" %}selected{% endif %}>
TensorFlow
</option>
</select>

<div class="buttons">
<button class="predict" type="submit">Predict</button>
<button class="reset" type="button"
onclick="window.location='/'">Reset</button>
</div>

</form>

<div class="samplebox">
<b>Available Test Samples:</b><br><br>
{% for row in samples %}
{{loop.index}} → {{row}}<br>
{% endfor %}
</div>

{% if result %}
<div class="result">
{{result|safe}}
</div>
{% endif %}

{% if chart %}
<div class="chartbox">
<img src="data:image/png;base64,{{chart}}" width="100%">
</div>
{% endif %}

</div>

</body>
</html>
"""

# ===============================================================
# 8. ROUTE
# ===============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    vals = {
        "sample_id": "",
        "model_choice": "tf"
    }

    result = ""
    chart = ""

    if request.method == "POST":

        vals = request.form.to_dict()

        sample_id = int(vals["sample_id"])
        model_choice = vals["model_choice"]

        row = np.array([test_samples[sample_id]])

        row_s = scaler.transform(row)

        if model_choice == "sk":
            pred = sk_model.predict(row_s)[0]
            used = "Scikit-Learn (Old Model)"
        else:
            pred = tf_model.predict(row_s, verbose=0)[0]
            used = "TensorFlow (Updated Model)"

        ent = pred[0]
        dist = pred[1]

        result = f"""
        <b>Selected Sample:</b> {sample_id+1}<br><br>
        <b>Model Used:</b> {used}<br><br>
        <b>Predicted Entropy:</b> {ent:.2f}<br>
        <b>Predicted Distance:</b> {dist:.2f} Mega Units
        """

        chart = build_chart(ent, dist)

    return render_template_string(
        HTML,
        vals=vals,
        result=result,
        chart=chart,
        samples=test_samples
    )

# ===============================================================
# 9. MAIN
# ===============================================================

if __name__ == "__main__":
    app.run(debug=True)