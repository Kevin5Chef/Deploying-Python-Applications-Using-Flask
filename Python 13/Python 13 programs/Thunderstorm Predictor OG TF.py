import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
print("SY-5, Kevin Victor, Roll No.-30")

# =========================================================
# 1. DATASET GENERATION (PHYSICS-INFORMED)
# =========================================================

np.random.seed(42)
n_samples = 1200

# Base features
temperature = np.random.uniform(25, 42, n_samples)
humidity = np.random.uniform(40, 100, n_samples)
pressure = np.random.uniform(990, 1020, n_samples)
uhi = np.random.uniform(0, 5, n_samples)
vertical_mixing = np.random.uniform(0, 10, n_samples)
moisture_inflow = np.random.uniform(0, 10, n_samples)

# =========================================================
# 2. FEATURE WEIGHTING (VERY IMPORTANT)
# =========================================================
# Boost key drivers of thunderstorms

temperature = temperature * 1.3
humidity = humidity * 1.3

# =========================================================
# 3. FEATURE ENGINEERING (CRITICAL)
# =========================================================

# Heat + moisture interaction (core thunderstorm driver)
heat_moisture_index = temperature * humidity / 100

# Instability proxy (rising warm air)
instability_index = vertical_mixing * temperature / 10

# Combine features
X = np.column_stack([
    temperature,
    humidity,
    pressure,
    uhi,
    vertical_mixing,
    moisture_inflow,
    heat_moisture_index,
    instability_index
])

# =========================================================
# 4. TARGET (IMPROVED REAL-WORLD LOGIC)
# =========================================================

y = (
    (
        (temperature > 30) &
        (humidity > 65) &
        (heat_moisture_index > 20) &
        (vertical_mixing > 4)
    )
    |
    (moisture_inflow > 6)
).astype(int)

# =========================================================
# 5. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 6. PREPROCESSING (NORMALIZATION)
# =========================================================

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-6

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# =========================================================
# 7. MODEL BUILDING (KERAS)
# =========================================================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# =========================================================
# 8. MODEL COMPILATION
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# 9. TRAINING
# =========================================================

history = model.fit(
    X_train, y_train,
    epochs=18,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
# =========================================================
# SAVE TRAINED MODEL AS .PKL FILE
# =========================================================

model_data = {
    "model": model,
    "mean": mean,
    "std": std
}

with open("thunderstorm_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("\nModel saved successfully as thunderstorm_model.pkl")
# =========================================================
# 10. EVALUATION
# =========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\n===== TEST PERFORMANCE =====")
print("Loss:", round(loss, 4))
print("Accuracy:", round(accuracy, 4))

# =========================================================
# 11. TEST CONDITIONS (REALISTIC SCENARIOS)
# =========================================================

test_conditions = np.array([
    [35, 85, 1005, 3, 8, 9],   # strong storm
    [30, 60, 1010, 2, 3, 2],   # weak
    [33, 75, 1008, 4, 6, 7],   # should trigger
    [28, 90, 1002, 1, 5, 6],   # borderline
    [37, 65, 995, 5, 9, 4]     # unstable but dry inflow
])

# Apply SAME feature transformations
temp = test_conditions[:, 0] * 1.3
hum = test_conditions[:, 1] * 1.3
press = test_conditions[:, 2]
uhi = test_conditions[:, 3]
vmix = test_conditions[:, 4]
minflow = test_conditions[:, 5]

heat_moisture = temp * hum / 100
instability = vmix * temp / 10

test_features = np.column_stack([
    temp, hum, press, uhi, vmix, minflow,
    heat_moisture, instability
])

# Normalize
test_features = (test_features - mean) / std

predictions = model.predict(test_features)

# =========================================================
# 12. OUTPUT RESULTS
# =========================================================

print("\n===== SAMPLE TEST RESULTS =====")

for i in range(len(test_conditions)):
    prob = predictions[i][0]
    pred = 1 if prob >= 0.5 else 0

    # Actual logic
    t = temp[i]
    h = hum[i]
    hm = heat_moisture[i]
    vm = vmix[i]
    mi = minflow[i]

    actual = (
        ((t > 30) and (h > 65) and (hm > 20) and (vm > 4))
        or (mi > 6)
    )

    print(f"\nCondition {i+1}: {test_conditions[i]}")
    print("Predicted Probability:", round(prob, 4))
    print("Predicted:", pred)
    print("Actual:", int(actual))

# =========================================================
# 13. EXPLANATION
# =========================================================

print("\n===== EXPLANATION =====")

print("""
Key Improvements:

1. Feature Weighting:
   - Temperature and humidity amplified (core drivers)

2. Feature Engineering:
   - Heat-Moisture Index (storm fuel)
   - Instability Index (rising air)

3. Better Target Logic:
   - Combines multiple atmospheric triggers

4. Model Learning:
   - Neural network captures non-linear interactions

Result:
Model now aligns better with real-world thunderstorm physics.
""")

import pickle

model_data = {
    "model": model,
    "mean": mean,
    "std": std
}

with open("thunderstorm_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("Saved as thunderstorm_model.pkl")