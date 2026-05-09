# =========================================================
# LOAD PRE-TRAINED MODEL (.PKL) USING PICKLE
# THUNDERSTORM PREDICTOR - FULL CORRECTED VERSION
#
# This version supports:
# 1. Loading dictionary-style saved file:
#    {
#       "model": trained_model,
#       "mean": mean,
#       "std": std
#    }
#
# 2. Background health checks
# 3. Timing model load
# 4. Threaded monitoring
# 5. User weather inputs
# 6. Proper normalization before prediction
# =========================================================

import pickle
import threading
import time
import os
import numpy as np

print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. GLOBAL SETTINGS
# =========================================================

MODEL_PATH = "thunderstorm_model.pkl"

model = None
mean = None
std = None

model_loaded = False
system_running = True


# =========================================================
# 2. HEALTH MONITOR THREAD
# =========================================================
def health_monitor():

    waiting_printed = False
    ready_printed = False

    while system_running:

        # Print waiting message only once
        if not model_loaded and not waiting_printed:
            print("[HEALTH CHECK] Waiting for model...")
            waiting_printed = True

        # Print ready message only once
        elif model_loaded and not ready_printed:
            print("[HEALTH CHECK] Model ready.")
            ready_printed = True

            # Stop monitoring after success
            break

        time.sleep(1)


# =========================================================
# 3. SAFE MODEL LOADER
# =========================================================

def load_model():

    global model, mean, std, model_loaded

    print("\nLoading pre-trained model...")

    start_time = time.time()

    # Check file exists
    if not os.path.exists(MODEL_PATH):
        print("ERROR: thunderstorm_model.pkl not found.")
        return

    try:
        with open(MODEL_PATH, "rb") as file:

            # ---------------------------------------------
            # LOAD SAVED DICTIONARY
            # ---------------------------------------------
            saved = pickle.load(file)

            model = saved["model"]
            mean = saved["mean"]
            std = saved["std"]

        model_loaded = True

        end_time = time.time()

        print("Model loaded successfully.")
        print("Load Time:", round(end_time - start_time, 4), "seconds")

    except Exception as e:
        print("Loading failed:", e)


# =========================================================
# 4. PREPROCESS INPUT
# Must match training pipeline exactly
# =========================================================

def preprocess_input(temp, hum, press, uhi, vmix, minflow):

    # ---------------------------------------------
    # SAME FEATURE WEIGHTING
    # ---------------------------------------------
    temp = temp * 1.3
    hum = hum * 1.3

    # ---------------------------------------------
    # SAME FEATURE ENGINEERING
    # ---------------------------------------------
    heat_moisture = temp * hum / 100
    instability = vmix * temp / 10

    X = np.array([[
        temp,
        hum,
        press,
        uhi,
        vmix,
        minflow,
        heat_moisture,
        instability
    ]])

    # ---------------------------------------------
    # SAME NORMALIZATION AS TRAINING
    # ---------------------------------------------
    X = (X - mean) / std

    return X


# =========================================================
# 5. PREDICTION LOOP
# =========================================================

def prediction_loop():

    while True:

        if not model_loaded:
            print("Model not ready yet...")
            time.sleep(2)
            continue

        try:
            print("\n===== ENTER WEATHER CONDITIONS =====")

            temp = float(input("Temperature: "))
            hum = float(input("Humidity: "))
            press = float(input("Pressure: "))
            uhi = float(input("Urban Heat Index: "))
            vmix = float(input("Vertical Mixing: "))
            minflow = float(input("Moisture Inflow: "))

            X = preprocess_input(temp, hum, press, uhi, vmix, minflow)

            start_pred = time.time()

            # -----------------------------------------
            # PREDICT
            # -----------------------------------------
            prob = model.predict(X, verbose=0)[0][0]

            end_pred = time.time()

            pred = 1 if prob >= 0.5 else 0

            # -----------------------------------------
            # OUTPUT
            # -----------------------------------------
            print("\n===== RESULT =====")
            print("Thunderstorm Probability:", round(float(prob), 4))
            print("Prediction:", pred)

            if pred == 1:
                print("Storm Likely")
            else:
                print("Storm Unlikely")

            print("Inference Time:",
                  round(end_pred - start_pred, 4),
                  "seconds")

        except KeyboardInterrupt:
            print("\nProgram stopped by user.")
            break

        except Exception as e:
            print("Input Error:", e)


# =========================================================
# 6. MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    # Start background monitor
    monitor_thread = threading.Thread(
        target=health_monitor,
        daemon=True
    )
    monitor_thread.start()

    # Load trained model
    load_model()

    # Start user predictions
    prediction_loop()

    system_running = False