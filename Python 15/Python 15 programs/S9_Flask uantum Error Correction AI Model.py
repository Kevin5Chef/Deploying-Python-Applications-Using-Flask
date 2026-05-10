# ===============================================================
# QUANTUM ERROR CORRECTION AI MODEL
# INPUT VALIDATION IN DEPLOYED APP
# ---------------------------------------------------------------
# FEATURES:
# 1. TensorFlow classification model
# 2. Synthetic quantum-state dataset (400 samples)
# 3. Error-corrects noisy quantum states into 0 / 1
# 4. Strong frontend input validation
# 5. 5 predefined test samples only
# 6. Reset button
# 7. Glassmorphism futuristic UI
# ===============================================================

print("SY-5, Kevin Victor, Roll No.-30")

from flask import Flask, request, render_template_string
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ===============================================================
# 1. DATASET CREATION (400 QUANTUM STATES)
# ---------------------------------------------------------------
# Features:
# amplitude_x       : real amplitude component
# amplitude_y       : imaginary amplitude component
# phase_angle       : radians
# decoherence       : noise level
# bit_flip_prob     : X-error probability
# phase_flip_prob   : Z-error probability
# interference      : constructive/destructive strength
# stabilizer_score  : parity-check confidence
# ---------------------------------------------------------------
# Target:
# logical_bit -> corrected output 0 or 1
# ===============================================================

np.random.seed(42)
samples = 400

amp_x = np.random.uniform(-1, 1, samples)
amp_y = np.random.uniform(-1, 1, samples)
phase = np.random.uniform(0, 2*np.pi, samples)
decoh = np.random.uniform(0, 1, samples)
bit_flip = np.random.uniform(0, 0.5, samples)
phase_flip = np.random.uniform(0, 0.5, samples)
interference = np.random.uniform(-1, 1, samples)
stabilizer = np.random.uniform(0, 1, samples)

# ===============================================================
# FEATURE ENGINEERING
# ===============================================================

magnitude = np.sqrt(amp_x**2 + amp_y**2)
noise_total = decoh + bit_flip + phase_flip
phase_wave = np.cos(phase)

X = np.column_stack([
    amp_x,
    amp_y,
    phase,
    decoh,
    bit_flip,
    phase_flip,
    interference,
    stabilizer,
    magnitude,
    noise_total,
    phase_wave
])

# ===============================================================
# TARGET LOGIC (SIMULATED ERROR CORRECTION)
# ---------------------------------------------------------------
# If signal strength + stabilizer dominate noise -> 1
# else -> 0
# ===============================================================

score = (
    1.5*magnitude +
    1.2*stabilizer +
    0.8*phase_wave +
    0.6*interference -
    1.8*noise_total
)

y = (score > 0.8).astype(int)

# ===============================================================
# 2. TRAIN TEST SPLIT
# ===============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================================
# 3. NORMALIZATION
# ===============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================================================
# 4. MODEL BUILDING
# ===============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(11,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ===============================================================
# 5. COMPILE
# ===============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.003),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================================================
# 6. TRAIN
# ===============================================================

model.fit(
    X_train,
    y_train,
    epochs=28,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ===============================================================
# 7. EVALUATE
# ===============================================================

loss, acc = model.evaluate(X_test, y_test, verbose=0)

print("\n===== MODEL READY =====")
print("Test Accuracy:", round(acc, 4))

# ===============================================================
# 8. PREDEFINED TEST SAMPLES
# ---------------------------------------------------------------
# 8 raw input features only
# ===============================================================

test_samples = [
    [0.9, 0.2, 0.5, 0.05, 0.02, 0.01, 0.7, 0.95],
    [0.2, 0.1, 3.0, 0.7, 0.35, 0.30, -0.6, 0.20],
    [0.6, 0.6, 1.2, 0.15, 0.08, 0.05, 0.5, 0.88],
    [0.1, 0.3, 5.5, 0.8, 0.40, 0.45, -0.7, 0.10],
    [0.75, 0.4, 2.1, 0.20, 0.10, 0.08, 0.45, 0.82]
]

# ===============================================================
# 9. FEATURE PREP FUNCTION
# ===============================================================

def prepare_features(row):
    ax, ay, ph, dc, bf, pf, inter, stab = row

    mag = np.sqrt(ax**2 + ay**2)
    nt = dc + bf + pf
    pw = np.cos(ph)

    data = np.array([[
        ax, ay, ph, dc, bf, pf, inter, stab,
        mag, nt, pw
    ]])

    return scaler.transform(data)

# ===============================================================
# 10. EXPLANATION GENERATOR
# ===============================================================

def explain(prob, sample):
    _, _, _, dc, bf, pf, inter, stab = sample

    if prob >= 0.75:
        return (
            "Strong stabilizer syndrome detected. "
            "Low decoherence and low flip probability. "
            "Parity matrix converged to logical state 1."
        )

    elif prob <= 0.25:
        return (
            "Noise operators dominate signal space. "
            "High decoherence disrupted basis alignment. "
            "Projection collapsed toward logical state 0."
        )

    else:
        return (
            "Mixed syndrome region detected. "
            "Competing amplitudes required probabilistic decoding. "
            "Nearest decision boundary selected corrected state."
        )

# ===============================================================
# 11. HTML
# ===============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Quantum Error Correction AI</title>

<style>
body{
margin:0;
font-family:Arial;
background:
radial-gradient(circle at top left,#00ffff22,transparent 30%),
radial-gradient(circle at bottom right,#8b5cf622,transparent 30%),
linear-gradient(135deg,#020617,#0f172a,#020617);
color:white;
min-height:100vh;
padding:30px;
}

.box{
max-width:900px;
margin:auto;
padding:30px;
border-radius:24px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(18px);
box-shadow:0 0 30px rgba(0,255,255,0.12);
animation:fade 1s ease;
}

@keyframes fade{
from{opacity:0;transform:translateY(30px);}
to{opacity:1;transform:translateY(0);}
}

h1{text-align:center;}

select{
width:100%;
padding:14px;
font-size:15px;
border:none;
border-radius:12px;
margin-top:14px;
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

.ok{
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
line-height:1.7;
}

.samplebox{
margin-top:20px;
padding:16px;
border-radius:16px;
background:rgba(255,255,255,0.05);
line-height:1.7;
}

.error{
margin-top:20px;
padding:15px;
background:#dc2626;
border-radius:14px;
}

small{opacity:0.8;}
</style>
</head>

<body>
<div class="box">

<h1>⚛ Quantum Error Correction AI</h1>

<form method="POST">

<select name="sample_id" required>
<option value="">Select Quantum Test Sample</option>
{% for i in range(5) %}
<option value="{{i}}" {% if vals.sample_id==i|string %}selected{% endif %}>
Sample {{i+1}}
</option>
{% endfor %}
</select>

<div class="buttons">
<button class="ok" type="submit">Correct State</button>
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

{% if error %}
<div class="error">{{error}}</div>
{% endif %}

{% if result %}
<div class="result">
{{result|safe}}
</div>
{% endif %}

<small>Logical state recovery using stabilizer-style decoding.</small>

</div>
</body>
</html>
"""

# ===============================================================
# 12. ROUTE WITH INPUT VALIDATION
# ===============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    vals = {"sample_id": ""}
    result = ""
    error = ""

    if request.method == "POST":

        vals = request.form.to_dict()

        # -------------------------------------------------------
        # VALIDATION CHECK 1: sample_id exists
        # -------------------------------------------------------
        sample_id = vals.get("sample_id", "").strip()

        if sample_id == "":
            error = "Please select a test sample."

        # -------------------------------------------------------
        # VALIDATION CHECK 2: numeric integer
        # -------------------------------------------------------
        elif not sample_id.isdigit():
            error = "Invalid sample format."

        else:
            idx = int(sample_id)

            # ---------------------------------------------------
            # VALIDATION CHECK 3: range check
            # ---------------------------------------------------
            if idx < 0 or idx >= len(test_samples):
                error = "Selected sample is out of range."

            else:
                sample = test_samples[idx]

                Xp = prepare_features(sample)

                prob = model.predict(Xp, verbose=0)[0][0]

                pred = 1 if prob >= 0.5 else 0

                explanation = explain(prob, sample)

                result = f"""
                <b>Selected Sample:</b> {idx+1}<br><br>
                <b>Corrected Logical Bit:</b> {pred}<br>
                <b>Confidence:</b> {prob:.4f}<br><br>
                <b>Mathematical Logic:</b><br>
                {explanation}
                """

    return render_template_string(
        HTML,
        vals=vals,
        result=result,
        error=error,
        samples=test_samples
    )

# ===============================================================
# 13. MAIN
# ===============================================================

if __name__ == "__main__":
    app.run(debug=True)