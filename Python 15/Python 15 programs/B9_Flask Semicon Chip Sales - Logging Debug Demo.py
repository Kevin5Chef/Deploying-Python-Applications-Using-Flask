# ===============================================================
# CHIP SALES PREDICTOR (INTENTIONALLY BROKEN + LOGGING DEMO)
# PURPOSE:
# Demonstrate how logs help detect/fix deployment problems
#
# Includes deliberate issues:
# 1. Missing values in dataset
# 2. Wrong datatypes
# 3. Exploding learning rate
# 4. Bad frontend CSS overlay
# 5. Chart render failures
# 6. Invalid user input
# 7. Prediction fallback errors
# 8. Template rendering warnings
#
# Logs clearly explain every issue on terminal.
# ===============================================================
print("SY-5, Kevin Victor, Roll No.-30")
from flask import Flask, request, render_template_string
import logging
import traceback
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
import os

print("SY-5, Kevin Victor, Roll No.-30")

# ===============================================================
# LOGGING CONFIGURATION
# ===============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

logger.info("System boot initiated.")

# ===============================================================
# FLASK APP
# ===============================================================

app = Flask(__name__)

# ===============================================================
# STEP 1: DATASET CREATION (WITH DELIBERATE ERRORS)
# ===============================================================

def build_dataset():
    try:
        logger.info("Building synthetic chip sales dataset.")

        np.random.seed(42)
        n = 500

        gen = np.random.randint(1, 5, n)
        phone_type = np.random.randint(0, 3, n)
        market = np.random.uniform(0, 10, n)
        interest = np.random.uniform(1, 10, n)
        services = np.random.uniform(1, 10, n)
        battery = np.random.uniform(1, 10, n)

        X = np.column_stack([
            gen, phone_type, market,
            interest, services, battery
        ])

        # Intentional corruption
        X[5][2] = np.nan
        X[8][4] = np.nan

        y = (
            gen * 20 +
            interest * 15 +
            services * 10 +
            market * 8 +
            battery * 5 +
            np.random.normal(0, 10, n)
        )

        logger.warning("Dataset contains intentional NaN values.")

        return X, y

    except Exception as e:
        logger.error("Dataset build failed.")
        logger.error(str(e))
        return None, None


# ===============================================================
# STEP 2: TRAIN MODEL (WITH DELIBERATE ISSUES)
# ===============================================================

def train_model():
    try:
        X, y = build_dataset()

        logger.info("Cleaning NaN values.")

        # Fix NaN
        X = np.nan_to_num(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logger.info("Building TensorFlow model.")

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu", input_shape=(6,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        # Intentional unstable LR
        logger.warning("Using intentionally high learning rate.")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss="mse",
            metrics=["mae"]
        )

        logger.info("Training started.")

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            verbose=1,
            validation_split=0.2
        )

        loss, mae = model.evaluate(X_test, y_test, verbose=0)

        logger.info(f"Training complete | Loss={loss:.3f} | MAE={mae:.3f}")

        return model, scaler

    except Exception as e:
        logger.error("Training crashed.")
        logger.error(traceback.format_exc())
        return None, None


# ===============================================================
# TRAIN ON STARTUP
# ===============================================================

model, scaler = train_model()

# ===============================================================
# HTML TEMPLATE (INTENTIONALLY BAD CSS OVERLAY)
# ===============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Broken Chip Sales Predictor</title>

<style>
body{
    margin:0;
    font-family:Arial;
    background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color:white;
    min-height:100vh;
}

/* Intentional ugly overlay */
.overlay{
    position:fixed;
    top:0;
    left:0;
    width:100%;
    height:100%;
    background:rgba(255,0,0,0.15);
    z-index:-1;
}

.card{
    width:700px;
    margin:auto;
    margin-top:40px;
    padding:30px;
    border-radius:25px;
    background:rgba(255,255,255,0.12);
    backdrop-filter:blur(12px);
    box-shadow:0 0 25px rgba(255,255,255,0.1);
}

input{
    width:95%;
    padding:10px;
    margin:8px;
    border-radius:10px;
    border:none;
}

button{
    padding:12px 18px;
    margin:10px;
    border:none;
    border-radius:10px;
    cursor:pointer;
    font-weight:bold;
}

.predict{background:#00ffaa;}
.reset{background:#ff5577;color:white;}

.chart-box{
    background:white;
    padding:20px;
    border-radius:20px;
    margin-top:20px;
}

.error{
    color:#ffaaaa;
    font-weight:bold;
}
</style>
</head>

<body>
<div class="overlay"></div>

<div class="card">
<h1>⚠ Broken Chip Sales Predictor</h1>
<p>Logs on terminal explain all issues transparently.</p>

<form method="POST">
<input name="gen" placeholder="Processor Generation (1-4)" value="{{vals.gen}}">
<input name="ptype" placeholder="Phone Type (0-2)" value="{{vals.ptype}}">
<input name="market" placeholder="Market Scenario" value="{{vals.market}}">
<input name="interest" placeholder="User Interest" value="{{vals.interest}}">
<input name="services" placeholder="Extendable Services" value="{{vals.services}}">
<input name="battery" placeholder="Battery Optimization" value="{{vals.battery}}">

<button class="predict">Predict</button>
<button class="reset" type="button" onclick="window.location='/'">Reset</button>
</form>

{% if msg %}
<h2>{{msg}}</h2>
{% endif %}

{% if error %}
<div class="error">{{error}}</div>
{% endif %}

{% if chart %}
<div class="chart-box">
<img src="data:image/png;base64,{{chart}}" width="100%">
</div>
{% endif %}

</div>
</body>
</html>
"""

# ===============================================================
# ROUTE
# ===============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    vals = {
        "gen":"",
        "ptype":"",
        "market":"",
        "interest":"",
        "services":"",
        "battery":""
    }

    msg = ""
    chart = None
    error = ""

    if request.method == "POST":

        try:
            logger.info("Prediction request received.")

            for k in vals:
                vals[k] = request.form[k]

            x = np.array([[
                float(vals["gen"]),
                float(vals["ptype"]),
                float(vals["market"]),
                float(vals["interest"]),
                float(vals["services"]),
                float(vals["battery"])
            ]])

            logger.info(f"Raw Input: {x}")

            x = scaler.transform(x)

            pred = model.predict(x, verbose=0)[0][0]

            msg = f"Predicted Monthly Chip Sales: {round(float(pred),2)} K Units"

            logger.info("Prediction successful.")

            # -------------------------------------------------------
            # GRAPH GENERATION
            # -------------------------------------------------------

            try:
                plt.figure(figsize=(7,4))
                features = [
                    "Gen","Type","Market",
                    "Interest","Services","Battery"
                ]
                vals_num = x[0]

                bars = plt.bar(features, vals_num)

                plt.grid(True, alpha=0.3)
                plt.title("Normalized Input Features")

                # Gradient simulation
                for b in bars:
                    b.set_alpha(0.8)

                img = io.BytesIO()
                plt.tight_layout()
                plt.savefig(img, format="png")
                img.seek(0)

                chart = base64.b64encode(img.getvalue()).decode()

                plt.close()

                logger.info("Chart rendered successfully.")

            except Exception as e:
                logger.error("Chart rendering failed.")
                logger.error(str(e))

        except ValueError:
            error = "Invalid numeric input detected."
            logger.warning("User entered invalid values.")

        except Exception as e:
            error = "Unexpected server issue."
            logger.error(traceback.format_exc())

    return render_template_string(
        HTML,
        vals=vals,
        msg=msg,
        chart=chart,
        error=error
    )


# ===============================================================
# HEALTH CHECKS
# ===============================================================

def startup_checks():
    logger.info("Running startup checks.")

    if model is None:
        logger.error("Model missing.")
    else:
        logger.info("Model loaded.")

    if scaler is None:
        logger.error("Scaler missing.")
    else:
        logger.info("Scaler ready.")

    logger.info("Startup checks complete.")


# ===============================================================
# MAIN
# ===============================================================

if __name__ == "__main__":

    startup_checks()

    logger.info("Flask server starting on http://127.0.0.1:5000")

    try:
        app.run(debug=True)

    except Exception as e:
        logger.critical("Flask startup failure.")
        logger.critical(traceback.format_exc())