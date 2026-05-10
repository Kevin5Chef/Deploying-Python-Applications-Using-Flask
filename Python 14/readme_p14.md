# Flask Full-Stack ML Applications — CSS Styling, In-Browser Charting, Regression Models, and Feature Contribution Visualisation
### A Technical Reference on Advanced Flask Frontends, Matplotlib Server-Side Rendering, Regression with TensorFlow, Scikit-learn Deployment, and Interactive ML Dashboards

**Author:** Kevin Victor
**Domain:** Python — Flask, TensorFlow, Scikit-learn, Matplotlib, Regression, Data Visualisation, Full-Stack ML Deployment
**Status:** Demonstrative & Applied

---

## Overview

This collection of Python programs represents the most complete form of the Flask-based ML deployment pipeline encountered so far: applications that simultaneously train predictive models, construct styled browser interfaces, generate data visualisations server-side, and expose interactive prediction endpoints — all within a single cohesive Python file. The programs move beyond basic API design into the domain of full-stack applied ML, where the engineering challenge is not merely training a model or building a server, but integrating both into a coherent, user-accessible product.

The implementations span six programs across three laboratory contexts. The Bucket programs (B6, B7) address the front-end layer of Flask development: advanced CSS styling with animated glassmorphism interfaces and quantum-themed visual design (B6), and server-side chart generation using Matplotlib with base64 image embedding for browser display (B7). The Scenario programs (S2, S4) address complete regression deployment: a house price prediction interface trained on an Indian real estate dataset with locality encoding (S2), and a semiconductor chip sales forecasting dashboard with side-by-side prediction and analytics chart panels (S4). The Experiment program (P14) consolidates the feature contribution visualisation workflow — training a scikit-learn Linear Regression model, computing coefficient-weighted feature contributions, and rendering a horizontal bar chart that explains each prediction in terms of the input features that drove it.

The central objective of this document is to explain what each component of these full-stack applications does, why each design decision is made, how server-side image generation integrates with browser rendering, how regression models differ from classifiers in their output and loss functions, and how feature contribution visualisation connects to the broader discipline of explainable AI.

---

## Context and Purpose

The programs in Practicals 12 and 13 established two distinct competencies: training neural networks for classification (P12), and deploying trained models as Flask API services (P13). Practical 14 unifies these competencies and extends them in two new directions.

The first extension is **visual output**. All previous programs returned predictions as plain text or JSON. The programs in this collection return rendered images — bar charts, horizontal contribution plots, and analytical dashboards — embedded directly in the browser page. This requires a new engineering pattern: generating matplotlib figures without a display screen, encoding them as base64 strings, and injecting those strings into HTML image tags. Understanding this pattern is essential for any data science web application that must communicate quantitative results visually.

The second extension is **regression**. All previous ML programs in this curriculum predicted discrete class labels. The programs in this collection predict continuous numerical values — house prices in lakhs, chip sales in thousands of units, rainfall probability as a decimal — using Mean Squared Error loss and reporting Mean Absolute Error as the evaluation metric. This shift from classification to regression changes the model's output layer, loss function, and the interpretation of predictions throughout the application.

The programs in this collection address the following engineering questions:

- What is `matplotlib.use("Agg")`, and why is it an absolute requirement for server-side chart rendering in Flask?
- How is a matplotlib figure converted to a base64-encoded PNG and embedded in an HTML `<img>` tag without writing any file to disk?
- What is the difference between Mean Squared Error and Mean Absolute Error, and why are both reported for regression models?
- What is `LabelEncoder`, and why is it necessary when categorical features such as locality names must be passed to a neural network?
- What does `model.predict()` return for a regression model with a single linear output neuron, and how is that value formatted for currency display?
- What is a coefficient-based feature contribution, and how does it differ from a full SHAP analysis?
- How does the Jinja2 `for` loop enable dynamic HTML form generation from a Python list?
- What is `validation_split` in `model.fit()`, and how does it differ from providing an explicit `validation_data` argument?

---

## Part I — CSS Styling and Animated Flask Interfaces

### 1. Advanced CSS in Flask — Structure, Animation, and Glassmorphism

The Quantum AI Landing Page (B6) demonstrates that Flask is not merely an API framework but a complete web serving platform capable of delivering production-quality, visually sophisticated browser interfaces. The program serves a single animated HTML page through `render_template_string()`, with all styling defined inline using advanced CSS techniques.

**CSS Grid and Flexbox** provide the layout foundation. The concept cards are arranged using `display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr))` — a responsive grid that automatically adjusts the number of columns based on the available viewport width. When the screen is wide, four cards appear in a row; on mobile screens, they collapse to a single column. This single CSS declaration replaces the manual breakpoint logic that older layout frameworks required.

**CSS custom animations** (`@keyframes`) define reusable motion sequences that CSS applies to elements automatically. The programs use several distinct animation patterns:

```css
@keyframes floatParticle {
    from { transform: translateY(105vh) scale(0.5); }
    to   { transform: translateY(-15vh) scale(1.3); }
}

@keyframes rise {
    from { opacity: 0; transform: translateY(60px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes blink {
    0%, 45%, 55%, 100% { transform: scaleY(1);   }
    50%                 { transform: scaleY(0.08); }
}
```

`floatParticle` moves six cyan dots from below the viewport to above it, creating a continuous upward drift that simulates quantum particle superposition. `rise` produces a smooth entry animation for the main glass panel — the card fades in while sliding up from 60 pixels below its final position. `blink` compresses the AI eye's pupil vertically to near-zero at the 50% keyframe, producing a lifelike blinking motion with natural easing at the hold points.

**Glassmorphism** is achieved through three CSS properties working in combination: `background: rgba(255,255,255,0.08)` creates a translucent white-tinted fill; `backdrop-filter: blur(18px)` applies a Gaussian blur to whatever is visible behind the element, creating the frosted-glass appearance; and `border: 1px solid rgba(255,255,255,0.18)` adds a faint white border that catches the light and reinforces the glass illusion. Together, these properties produce the characteristic semi-transparent card aesthetic that has become the dominant design language for modern AI and technology product interfaces.

**Gradient text** — used for the hero headline — applies a multi-stop linear gradient to the text fill rather than the background, achieved through the `-webkit-background-clip: text` and `-webkit-text-fill-color: transparent` combination. The gradient flows from cyan through purple to green, visually encoding the "convergence" concept that the page describes.

**Responsive design** with `@media` queries adjusts font sizes and layout on narrow screens: `@media(max-width: 768px) { .hero h1 { font-size: 42px; } }`. This ensures the landing page remains legible and usable on mobile devices without requiring a separate mobile codebase.

---

### 2. Server-Side Chart Rendering — Matplotlib, BytesIO, and Base64

The Flask Temperature Bar Chart program (B7) introduces the most important data visualisation pattern in Flask development: generating matplotlib figures on the server without any screen or display, encoding them as binary PNG data, and delivering them to the browser as base64 strings embedded directly in the HTML response.

The critical first step is the Matplotlib backend selection:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Matplotlib has multiple rendering backends, each designed for a different output context. The default backend (`TkAgg` on most systems) attempts to open a graphical display window when a figure is created. In a Flask server process running without a display — which is the standard configuration for any server, whether local or cloud-hosted — this attempt raises a `_tkinter.TclError: no display name` exception that crashes the server. The `Agg` backend renders figures to an in-memory raster image buffer without any display, making it the correct backend for all server-side rendering contexts. This single `matplotlib.use("Agg")` call must precede any import of `matplotlib.pyplot` — changing the backend after pyplot has been imported has no effect.

The server-side rendering pipeline proceeds through four steps:

```python
# Step 1: Create the figure and draw
plt.figure(figsize=(10, 5))
ax = plt.gca()
ax.bar([...], max_vals, ...)
ax.bar([...], min_vals, ...)

# Step 2: Write to an in-memory binary buffer
img = io.BytesIO()
plt.savefig(img, format="png", dpi=140, bbox_inches="tight")
plt.close()
img.seek(0)

# Step 3: Encode to base64
encoded = base64.b64encode(img.getvalue()).decode()

# Step 4: Embed in HTML
# <img src="data:image/png;base64,{{ chart }}">
```

`io.BytesIO()` creates an in-memory binary buffer — a file-like object that exists in RAM rather than on disk. `plt.savefig(img, format="png")` writes the rendered figure to this buffer in PNG format. `plt.close()` releases the figure's memory; this is critical in a long-running server where each request generates a new figure — without `plt.close()`, matplotlib retains all previous figures in memory until the server is restarted, causing a memory leak. `img.seek(0)` resets the buffer's read position to the beginning, so that `img.getvalue()` reads the entire PNG binary. `base64.b64encode()` converts the binary PNG bytes to a base64 ASCII string, which is safe to embed in HTML text. `.decode()` converts the resulting `bytes` object to a Python string.

The base64 string is injected into the HTML template using the **data URI scheme** — a standard that allows binary data to be embedded directly in HTML attributes as a URL:

```html
<img src="data:image/png;base64,{{ chart }}">
```

The browser decodes the base64 string back to binary PNG data and renders it as an image, exactly as it would render an image loaded from a URL. The result is a fully self-contained HTML response that includes the chart without requiring any file system access, any static file server, or any secondary HTTP request.

**Per-day colour gradients** in the bar chart are achieved by providing a list of hex colour strings — one per bar — to Matplotlib's `bar()` function. The warm gradient (from coral through amber) for maximum temperatures and the cool gradient (from light blue through deep blue) for minimum temperatures provide immediate visual encoding of the data's meaning: warmer colours for warmer values.

**Input validation in the chart route** catches the edge case where a day's minimum temperature exceeds its maximum, raising a domain-meaningful `ValueError` before passing values to matplotlib:

```python
if mn > mx:
    raise ValueError(f"Day {i+1}: Min cannot exceed Max.")
```

The error is caught at the route level and passed to the template as an `error` variable, displayed in a styled error banner without a server traceback.

---

## Part II — Regression Model Deployment

### 3. Regression vs Classification — Output, Loss, and Evaluation

All previous ML programs in this curriculum performed classification: the model's output was a discrete class label (safe/unsafe, positive/negative, Room 0/Room 1), produced by a sigmoid or softmax output layer trained with binary cross-entropy or categorical cross-entropy loss. The programs in this practical perform regression: the model predicts a continuous numerical value — a price in lakhs, a sales volume in thousands of units — produced by a single linear output neuron with no activation function, trained with Mean Squared Error loss.

The architectural difference is confined to the output layer:

```python
# Classification (previous practicals):
tf.keras.layers.Dense(2, activation="softmax")   # 2-class probabilities
tf.keras.layers.Dense(1, activation="sigmoid")   # Binary probability

# Regression (this practical):
tf.keras.layers.Dense(1)                          # Unbounded real number
```

A linear output neuron (no activation function) can produce any real number — positive, negative, very large, or very small — which is appropriate for price or sales volume prediction. Applying softmax or sigmoid to the output would artificially constrain the prediction to a range of [0, 1] or a probability distribution, which has no meaningful interpretation for a house price.

**Mean Squared Error (MSE)** is the standard regression loss function. For a prediction $\hat{y}$ and a true value $y$, MSE is $(\hat{y} - y)^2$. Squaring the error has two effects: it makes all errors positive (so over-predictions and under-predictions both contribute positively to the loss), and it penalises large errors disproportionately — an error of 20 lakhs contributes four times as much to the loss as an error of 10 lakhs. This quadratic penalty makes the model particularly sensitive to large outlier errors during training.

**Mean Absolute Error (MAE)** is reported alongside MSE as the evaluation metric because it is directly interpretable in the prediction's native units. If the house price model reports `Test MAE: 12.45 Lakhs`, this means the model's predictions are on average 12.45 lakhs away from the true price — a quantity that a domain expert can immediately assess for acceptability. MSE is used for training (its smooth gradient makes Adam's adaptive updates effective), while MAE is reported for human interpretation.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss="mse",
    metrics=["mae"]
)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print("Test MAE:", round(mae, 2), "Lakhs")
```

---

### 4. LabelEncoder — Encoding Categorical Features for Neural Networks

Neural networks require numerical inputs. Categorical features — such as locality names ("Baner", "Kothrud", "Wakad") or phone types ("gaming", "ai_powered", "high_performance") — must be converted to numerical form before being passed to a Dense layer. `LabelEncoder` from scikit-learn performs this conversion by assigning a unique integer to each unique category value.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["locality_enc"] = le.fit_transform(df["locality"])
# "Aundh" → 0, "Baner" → 1, "Hinjewadi" → 2, ...
```

`le.fit_transform(df["locality"])` scans all unique values in the column, sorts them alphabetically, and maps each to a consecutive integer starting at 0. The fitted encoder remembers this mapping, enabling the same transformation to be applied to new input values at inference time:

```python
loc_enc = le.transform([locality])[0]
```

**An important limitation of label encoding** is that it introduces an implicit ordinal relationship between categories that may not exist in reality. Encoding "Baner" as 1 and "Hinjewadi" as 2 implies that Hinjewadi is "greater than" Baner by one unit — a relationship that is arithmetically present in the encoding but semantically meaningless. For models that learn linear combinations of input features (such as the first Dense layer's weighted sum), this can mislead the model into treating geographical proximity in the integer encoding as geographical proximity in reality.

The technically correct solution for nominal categorical features is **one-hot encoding**, which represents each category as a binary vector with a 1 in exactly one position. For seven localities, each observation would be represented as a 7-element binary vector. One-hot encoding eliminates the spurious ordinal relationship by treating each category as an independent dimension. In the house price programs, label encoding is used as a practical approximation — sufficient for demonstration purposes given that the neural network's non-linear layers can partially compensate for the spurious ordinality through learned transformations.

The Sales Dashboard (S4) uses two separate encoders — `phone_encoder` and `market_encoder` — one for each categorical feature. Both are fitted on the training data and stored as module-level variables, so they are available to the Flask route function at inference time:

```python
phone_encoder = LabelEncoder()
market_encoder = LabelEncoder()

phone_enc = phone_encoder.fit_transform(phone_type)
market_enc = market_encoder.fit_transform(market_scenario)
```

Storing fitted encoders alongside the fitted scaler is the deployment-time equivalent of storing the `mean` and `std` normalisation parameters in the previous practical — all preprocessing state that was computed during training must be preserved and applied identically at inference time.

---

### 5. House Price Prediction — Dataset Design and Price Engineering

The House Price Prediction interface (S2) trains a regression model on a synthetic dataset of 2,500 properties, designed to reflect the structure of the Pune real estate market. The target price is constructed as an explicit function of the input features, with domain-appropriate coefficients:

```python
price = (
    area      * 0.012  +   # 600–4000 sq ft, primary driver
    bedrooms  * 18     +   # 2–5 BHK
    amenities * 3.5    +   # clubhouse, gym, pool count
    club      * 10     +   # binary: club membership premium
    np.array([loc_boost[x] for x in locality])  -  # locality premium
    city_dist * 1.2    -   # proximity penalty per km
    station_dist * 0.5 -
    airport_dist * 0.3 +
    np.random.normal(0, 8, samples)   # market noise
)
```

The locality boost dictionary encodes the premium associated with each area: Aundh commands the highest premium at 38 lakhs above baseline, reflecting its established residential status; Hinjewadi carries the lowest at 24, reflecting its IT-corridor character where properties are valued differently. This structured noise — Gaussian noise of standard deviation 8 lakhs — models market variability without obscuring the underlying pricing signal.

**Dynamic currency formatting** at inference time adapts the display between lakhs and crores based on the predicted value:

```python
if pred >= 100:
    result = f"₹ {pred/100:.2f} Crores"
else:
    result = f"₹ {pred:.2f} Lakhs"
```

100 lakhs equals 1 crore. This formatting logic ensures that high-value properties — large Aundh flats with extensive amenities — are displayed in the appropriate Indian currency denomination rather than as a number exceeding 100, which would be confusing in the lakhs scale.

**Retaining form values after prediction** is implemented by passing a `vals` dictionary back to the template, pre-populated with the submitted form values:

```python
if request.method == "POST":
    vals = request.form.to_dict()    # captures all submitted fields
    ...
    pred = model.predict(features)[0][0]
    result = f"₹ {pred:.2f} Lakhs"

return render_template_string(HTML, result=result, vals=vals, localities=localities)
```

The Jinja2 template uses `value="{{ vals.area }}"` on each input field to pre-fill the submitted value. Without this pattern, form fields reset to empty after every prediction, requiring the user to re-enter all values each time — a poor user experience that is trivially corrected by passing the submitted values back through the template context.

---

### 6. Jinja2 Dynamic Form Generation — Loops and Conditional Selection

The forms in the house price and sales dashboard programs use Jinja2 `for` loops to generate HTML `<select>` dropdown menus dynamically from Python lists, eliminating the need to hardcode `<option>` elements manually:

```html
<select name="bedrooms">
{% for b in [2, 2.5, 3, 3.5, 4, 4.5, 5] %}
    <option value="{{ b }}"
        {% if vals.bedrooms == b|string %}selected{% endif %}>
        {{ b }} BHK
    </option>
{% endfor %}
</select>
```

`{% for b in [2, 2.5, 3, 3.5, 4, 4.5, 5] %}` iterates over the bedrooms list, generating one `<option>` element per value. The `{% if vals.bedrooms == b|string %}selected{% endif %}` conditional adds the HTML `selected` attribute to the option that matches the previously submitted value — the `b|string` filter converts the numeric `b` to a string for comparison with the form value (which Flask always returns as a string). This combination of loop and conditional renders the correct option as pre-selected after form submission, maintaining state across requests without JavaScript.

The same pattern generates locality and phone type dropdowns from Python lists passed through the template context:

```python
return render_template_string(HTML, ..., localities=localities)
```

```html
{% for loc in localities %}
<option value="{{ loc }}" {% if vals.locality == loc %}selected{% endif %}>
    {{ loc }}
</option>
{% endfor %}
```

This approach — rendering HTML structure from Python data using Jinja2 loops — is the server-side rendering equivalent of JavaScript's `Array.map()` for generating DOM elements. It keeps the list of valid options defined once in the Python model code (where it is coupled to the `LabelEncoder`'s vocabulary) rather than duplicated in the HTML template.

---

### 7. validation_split vs validation_data in model.fit()

The regression programs use `validation_split=0.2` in `model.fit()` rather than the explicit `validation_data=(X_test, y_test)` pattern used in previous practicals:

```python
model.fit(
    X_train, y_train,
    epochs=35,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

`validation_split=0.2` instructs Keras to reserve 20% of the training data as an internal validation set, drawn from the end of `X_train`. This data is not used for weight updates — gradients are not computed on it — but the model is evaluated on it at the end of each epoch, producing `val_loss` and `val_mae` entries in the training history. This approach is convenient when the dataset is large enough that withholding 20% for validation does not meaningfully reduce the training set, and when a separate test set (`X_test`) has already been held out for final evaluation.

The distinction from `validation_data=(X_test, y_test)` is important: when `validation_split` is used, the validation data is drawn from the training array itself, not from the separately held-out test set. This means the test set remains entirely unseen during training — including during per-epoch validation — producing a clean final evaluation. Using `X_test` as `validation_data` during training technically exposes the test set to per-epoch monitoring, which can lead to inadvertent overfitting to the test distribution through hyperparameter choices.

---

### 8. Dropout in Regression Models

The Sales Dashboard model (S4) includes a `Dropout(0.15)` layer after the first Dense layer:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.15),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])
```

Dropout randomly sets 15% of the first hidden layer's activations to zero during each training step. This prevents any individual neuron from becoming an indispensable memoriser of specific training examples, encouraging the network to distribute the learning of each sales pattern across multiple neurons. The effect is a model that generalises better to new chip configurations not seen during training.

Dropout is inactive during `model.predict()` — Keras automatically sets all dropout layers to pass-through mode when the model is called outside of training, ensuring that inference predictions are deterministic and use the full trained capacity of the network. This behaviour is the reason `model.eval()` exists as a separate call in PyTorch (where the programmer controls the mode explicitly), while in Keras it is handled automatically based on whether the call is inside a training step.

A `Dropout(0.15)` rate is conservatively low — only 15% of neurons are zeroed per step. This reflects the fact that regression models trained on structured tabular data are generally less prone to severe overfitting than classification models trained on image data, where dropout rates of 0.3–0.5 are common. The 0.15 rate provides mild regularisation without substantially slowing convergence.

---

## Part III — Feature Contribution Visualisation

### 9. Coefficient-Based Feature Contributions — The Linear Model Advantage

The Rainfall Predictor experiment (P14) uses a scikit-learn `LinearRegression` model, which has a direct and mathematically transparent relationship between input features and output prediction. For a linear model, the prediction is:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_{10} x_{10} + b$$

where $w_i$ are the model's learned coefficients (accessible as `model.coef_`) and $x_i$ are the scaled input feature values. The contribution of feature $i$ to the prediction is simply $w_i \cdot x_i$ — the coefficient multiplied by the scaled feature value. Features with large positive contributions push the prediction toward higher rainfall probability; features with large negative contributions push it toward lower probability.

```python
def generate_plot(sample_scaled):
    coeffs  = model.coef_              # learned weights (10,)
    contrib = sample_scaled.flatten() * coeffs  # element-wise product

    plt.figure(figsize=(8, 4))
    plt.barh(FEATURE_NAMES, contrib)   # horizontal bar chart
    plt.title("Feature Contribution (Linear Model)")
    plt.grid(True, linestyle="--", alpha=0.5)
    ...
```

A **horizontal bar chart** is the appropriate visualisation for feature contributions because it uses the horizontal axis (the longer axis in landscape figures) for the magnitude of contribution — which is the quantity of interest — and the vertical axis for feature labels, which are most readable as horizontal text. Positive contributions extend to the right (rainfall-positive features); negative contributions extend to the left (rainfall-suppressing features). The chart is generated fresh for each prediction request, reflecting the specific feature values of the selected test sample.

**Why this is interpretable to a domain expert:** A meteorologist looking at the feature contribution chart for a high-rainfall sample can immediately see that high humidity and moisture inflow are the dominant positive contributors, while high atmospheric pressure is the dominant negative contributor — reflecting established meteorological knowledge about conditions that inhibit rainfall. The chart makes the model's reasoning legible without requiring the viewer to understand the internal mathematics of the regression.

**The relationship between coefficient-based contributions and SHAP:** The experimental specification references SHAP (SHapley Additive Explanations) as the target interpretability method. SHAP values generalise the coefficient-based approach to non-linear models: for a linear model, SHAP values are exactly equal to $w_i \cdot x_i$ (the contributions computed in this program). For non-linear models such as random forests or neural networks, SHAP values approximate the feature contributions using game-theoretic Shapley values, which are computed by evaluating the model across all possible subsets of features and averaging the marginal contribution of each feature across all orderings. The coefficient-based approach in P14 is a practical implementation of the same concept for the linear case, producing identical results to `shap.LinearExplainer` applied to a `LinearRegression` model.

---

### 10. Scikit-learn LinearRegression — Pipeline and Deployment

The Rainfall Predictor uses scikit-learn's `LinearRegression` rather than a TensorFlow neural network. This is an appropriate model choice for a problem with 10 continuous features and a continuous target in the range [0, 1] — the linear model's transparency is a feature, not a limitation, because it enables the coefficient-based feature contribution analysis that is the program's primary objective.

```python
from sklearn.linear_model import LinearRegression

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, rain_prob, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
```

`LinearRegression.fit()` solves the ordinary least squares problem analytically — it does not use gradient descent or require epochs, batch sizes, or a learning rate. The solution is computed in a single matrix operation, making training effectively instantaneous for tabular datasets of this size.

**Soft labels as regression targets:** The target variable `rain_prob` is constructed as a weighted sum of normalised features, producing continuous values in [0, 1] rather than binary 0/1 labels. This is the regression equivalent of the soft labelling strategy demonstrated in the NLP programs of Practical 12 — rather than forcing the model to predict a sharp binary outcome (rain or no rain), the soft target encodes the degree of rainfall likelihood, enabling the model to learn graded probability estimates. The threshold `prob >= 0.5` is applied at prediction time to produce a binary class label for display.

**`np.clip(prob, 0, 1)`** after prediction ensures the output stays within the valid probability range. A linear model can predict values outside [0, 1] for inputs that fall outside the training distribution — clipping prevents the display of nonsensical probabilities such as 1.3 or -0.2.

---

### 11. Sales Dashboard — Two-Panel Layout and Analytics Capsule

The Sales Dashboard (S4) demonstrates a more sophisticated frontend layout: a two-panel side-by-side grid that displays the input form on the left and the chart visualisation on the right simultaneously:

```css
.wrapper {
    max-width: 1100px;
    margin: auto;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    align-items: start;
}
```

`grid-template-columns: 1fr 1fr` divides the wrapper into two equal-width columns. The left panel (`class="card"`) contains the prediction form; the right panel (`class="graphcard"`) displays the analytics chart. `align-items: start` prevents the shorter panel from stretching to match the taller one — both panels grow with their content rather than being forced to equal height.

The chart generated for the sales dashboard is a vertical bar chart that compares the predicted sales volume against scaled versions of the five demand-driver input features, enabling the user to see not just the prediction value but how it relates to the input conditions that produced it:

```python
values = [
    pred,
    float(vals["user_interest"]) * 25,
    float(vals["services"])      * 25,
    float(vals["brand_power"])   * 25,
    float(vals["battery_eff"])   * 25,
    float(vals["price_comp"])    * 25
]
```

Multiplying the 1–10 scale input values by 25 maps them to a 25–250 range that is comparable in magnitude to the predicted sales values (which are typically in the range of 200–600 thousand units). This visual scaling — while not a rigorous normalisation — enables meaningful comparison between the prediction and the features on the same chart axis, providing the user with an intuitive sense of which inputs contributed most to the predicted sales volume.

---

## Part IV — Industrial Use Cases

### Use Case 1 — Real Estate Valuation Platforms (S2)

**Application Domain:** Property Technology (PropTech), Automated Valuation Models, Real Estate Analytics

The House Price Prediction web interface models the core functionality of automated valuation model (AVM) platforms deployed by real estate technology companies worldwide. MagicBricks, 99acres, NoBroker, and Housing.com — the leading Indian PropTech platforms — provide instant price estimates for listed properties using models trained on historical transaction data and property characteristics including area, locality, amenity score, and distance to key urban landmarks.

The eight input features in S2 — area, bedroom count, locality, amenity count, club membership, distance to city centre, station, and airport — are a representative subset of the features used in production real estate models. Production AVMs supplement these structural features with additional signals including: historical transaction prices in the locality (a lagged price series), macroeconomic indicators (RBI repo rate, housing loan interest rates), real-time demand signals (search query volumes on the platform), and satellite-derived features (green cover, road density, proximity to water bodies).

The Indian currency formatting logic — lakhs below 100, crores above — reflects the convention universally used in Indian real estate marketing materials and is a domain-specific engineering requirement that must be correctly implemented for the interface to be credible to Indian users.

---

### Use Case 2 — Semiconductor Sales Forecasting (S4)

**Application Domain:** Semiconductor Industry, Demand Forecasting, Supply Chain Planning

The Chip Sales Dashboard models the demand forecasting systems used by semiconductor manufacturers — companies such as Qualcomm, MediaTek, and Intel — to predict quarterly sales volumes for processor generations across different device categories and market conditions.

The four market scenarios — weak, normal, strong, boom — model the demand cycle that characterises the semiconductor industry. Semiconductor demand is cyclically volatile: inventory buildups during periods of strong demand are followed by sharp corrections as customers deplete excess stock. The market scenario feature captures this cycle-position effect, with the boom multiplier (+55 thousand units over the weak scenario's -20) reflecting the observed demand amplification that occurs during supply shortages.

The `perf_score` and `demand_score` engineered features:

```python
perf_score  = processor_gen * 2.2 + services * 0.8 + battery_eff * 0.6
demand_score = user_interest * 1.5 + brand_power * 1.2 + price_comp * 0.9
```

Model the two primary axes of product competitiveness in the smartphone processor market: technical performance (driven by processor generation and power efficiency) and market demand momentum (driven by consumer interest, brand strength, and competitive pricing). These composite features compress ten raw inputs into higher-level abstractions that are more directly related to sales volume, reducing the burden on the neural network to discover these relationships from scratch.

---

### Use Case 3 — Rainfall and Weather Nowcasting (P14)

**Application Domain:** Meteorology, Climate Services, Agricultural Planning, Disaster Management

The Rainfall Predictor models the short-range precipitation probability system used in operational weather services. The ten features span the full scale hierarchy of atmospheric dynamics: local thermodynamic conditions (temperature, humidity, surface pressure), meso-scale dynamics (moisture inflow, vertical mixing), synoptic-scale circulation features (jet stream, ocean current, ENSO index), and kinematic indicators (vorticity, wind shear).

Weather nowcasting systems — providing forecasts for the next 0–6 hours at high spatial resolution — are operated by national meteorological services including the India Meteorological Department (IMD), the UK Met Office, and the National Weather Service (NOAA). These systems increasingly use machine learning models as post-processing layers on top of numerical weather prediction model output, correcting systematic biases and providing calibrated probability estimates for local precipitation.

The feature contribution chart is particularly valuable in this domain because meteorologists need to understand not just whether rain is predicted but which atmospheric conditions are driving the prediction — enabling them to assess the model's physical plausibility and flag cases where the model may be extrapolating beyond its training distribution.

---

### Use Case 4 — Data Visualisation in Scientific Web Applications (B7)

**Application Domain:** Scientific Computing, Environmental Monitoring, Data Journalism

The server-side matplotlib rendering pattern demonstrated in the Temperature Bar Chart program (B7) is the standard architecture for scientific web applications that must display data visualisations generated from user inputs or dynamic data. Projects including Plotly Dash, Bokeh Server, Streamlit, and Shiny (R) all implement variations of this pattern — generating figures on the server in response to user inputs and returning rendered images or interactive plots to the browser.

The base64 embedding approach — while simple and dependency-free — is most appropriate for static images where the full chart can be regenerated on each request. Applications that require interactive charts (pan, zoom, hover tooltips) use client-side rendering libraries such as Plotly.js or Chart.js, which receive data as JSON from the server and render the chart in the browser using JavaScript. The server-side rendering and client-side rendering patterns address complementary use cases and are often combined in the same application.

---

## Part V — Future Scope and Industry-Grade Upgrade Paths

### 1. One-Hot Encoding — Eliminating Spurious Ordinality

The label encoding used in S2 and S4 can be replaced with one-hot encoding for production models:

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first", sparse_output=False),
     ["locality", "phone_type", "market_scenario"]),
    ("scale", StandardScaler(),
     ["area", "bedrooms", "amenities", ...])
])

X_processed = preprocessor.fit_transform(df)
```

`ColumnTransformer` applies different transformations to different columns — one-hot encoding to categorical features and standard scaling to numerical features — in a single pipeline step. `drop="first"` removes the first dummy variable from each encoded group to avoid multicollinearity. The fitted `ColumnTransformer` can be pickled alongside the model, ensuring consistent preprocessing at inference time. This is the production-standard approach for handling mixed categorical and numerical feature sets.

---

### 2. Interactive Charts — Plotly.js Integration

The static matplotlib charts embedded as base64 PNGs can be replaced with interactive Plotly charts that support hover tooltips, zooming, and dynamic updates:

```python
import plotly.graph_objects as go
import json

def create_interactive_chart(max_vals, min_vals):
    fig = go.Figure(data=[
        go.Bar(name="Max Temp", x=days, y=max_vals, marker_color="#ff6b6b"),
        go.Bar(name="Min Temp", x=days, y=min_vals, marker_color="#4dabf7")
    ])
    fig.update_layout(barmode="group", template="plotly_dark")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
```

The chart JSON is passed to the template and rendered client-side using Plotly.js loaded from a CDN:

```html
<div id="chart"></div>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
    const data = {{ chart_json | safe }};
    Plotly.newPlot("chart", data.data, data.layout);
</script>
```

Interactive charts enable users to hover over bars to see precise values, zoom into regions of interest, and toggle data series on and off — significantly improving the analytical utility of the dashboard compared to static images.

---

### 3. True SHAP Integration — Explaining Non-Linear Models

The coefficient-based feature contributions in P14 are exact for linear models but do not generalise to the neural network models used in S2 and S4. True SHAP explanations for neural networks are computed using `shap.DeepExplainer` or `shap.GradientExplainer`:

```python
import shap

# For TensorFlow models
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:5])

# Waterfall plot for a single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0][0],
    base_values=explainer.expected_value[0],
    data=X_test[0],
    feature_names=FEATURE_NAMES
))
```

SHAP waterfall plots show the base expected value of the model and the positive or negative contribution of each feature to the deviation of the actual prediction from that base value. This visualisation is the current industry standard for explaining individual predictions from complex models, adopted by regulatory frameworks for high-stakes decisions in finance (credit scoring), healthcare (diagnostic support), and hiring (resume screening).

---

### 4. Model Persistence — Saving Regression Models for Production

The regression models in S2 and S4 are retrained every time the Flask server starts. For models that take minutes to train, this is acceptable. For large models that take hours, the model should be trained once, saved, and loaded on startup:

```python
# Save after training
import joblib
joblib.dump({
    "model": model,      # TF model: use model.save() instead
    "scaler": scaler,
    "le": le
}, "house_price_model.joblib")

# Load on server startup
saved = joblib.load("house_price_model.joblib")
model  = saved["model"]
scaler = saved["scaler"]
le     = saved["le"]
```

For TensorFlow models, the native format is preferable:

```python
model.save("house_price_model.keras")
model = tf.keras.models.load_model("house_price_model.keras")
```

Saving the `LabelEncoder` alongside the model is as critical as saving the `StandardScaler` — the encoder's category-to-integer mapping must be identical between training and inference.

---

### 5. Caching Chart Generation — Reducing Server Load

Each prediction request in the current programs generates a fresh matplotlib figure, which involves significant CPU work (figure creation, rendering, PNG encoding, base64 encoding). Under concurrent load, this can become a bottleneck. Flask-Caching provides a simple memoisation layer:

```python
from flask_caching import Cache

cache = Cache(app, config={"CACHE_TYPE": "SimpleCache"})

@cache.memoize(timeout=300)
def generate_chart(max_vals, min_vals):
    ...
```

`@cache.memoize(timeout=300)` caches the function's return value for 300 seconds, keyed on the input arguments. Identical requests within 5 minutes receive the cached base64 string without re-generating the chart. For production deployments with many concurrent users submitting similar inputs, caching reduces both response latency and server CPU utilisation.

---

## Conclusion

The programs in this collection complete the core Flask ML deployment curriculum by adding the two capabilities that distinguish a functional ML demo from a usable data product: visual output and regression prediction. Server-side chart generation with matplotlib transforms raw numerical predictions into immediately comprehensible visual summaries. Regression models with continuous outputs broaden the class of problems that can be addressed beyond binary and multi-class classification. Feature contribution visualisation provides the interpretability layer that makes model predictions trustworthy and auditable for domain experts.

Taken together, the six programs in this practical demonstrate the full range of what a single Python file — combining a trained model, a Flask server, a styled HTML interface, and a chart renderer — can deliver to a browser. The house price predictor accepts property specifications and returns a price estimate in lakhs or crores; the sales dashboard accepts chip specifications and market conditions and returns a demand forecast alongside a comparison chart; the rainfall predictor accepts atmospheric measurements and returns a probability estimate alongside a horizontal bar chart that explains which features drove the prediction.

Each of these applications reflects a real class of deployed AI product, and each builds directly on the Flask, preprocessing, and model training foundations established in the preceding practicals. The upgrade paths — interactive charts with Plotly, true SHAP explanations for neural networks, one-hot encoding, model persistence, and caching — define the engineering investment that converts these functional prototypes into production-grade AI services.

---

## File Reference

| File | Core Concept | Domain |
|---|---|---|
| `B6_Flask Quantum AI Landing Page.py` | Advanced CSS Styling, Glassmorphism, Keyframe Animations, Gradient Text, CSS Grid, Responsive Design | Web Development / AI/Quantum Branding |
| `B7_Flask Temperature Bar Chart.py` | `matplotlib.use("Agg")`, Server-Side Chart Rendering, `io.BytesIO`, Base64 PNG Encoding, Data URI Embedding, Input Validation | Data Visualisation / Environmental Monitoring |
| `S2_House Price Prediction – Flask + Tensorflow.py` | TensorFlow Regression, `LabelEncoder`, `StandardScaler`, MSE/MAE Loss, Currency Formatting, Form State Retention, Jinja2 Dynamic Selects | Regression Deployment / Real Estate Valuation |
| `S4_Flask Sales Prediction Dashboard.py` | Regression with Dropout, Multi-Encoder Categorical Features, Feature Engineering, `validation_split`, Two-Panel CSS Grid Layout, Analytics Chart Capsule | Sales Forecasting / Semiconductor Demand |
| `P14_Rainfall Predictor – Flask + Sklearn + Matplotlib.py` | Scikit-learn Linear Regression, Soft Labels, Coefficient-Based Feature Contributions, Horizontal Bar Chart, SHAP Conceptual Foundation, JSON API with Embedded Plot | Feature Contribution Visualisation / Meteorology |

---

*"The goal is to turn data into information, and information into insight." — Carly Fiorina. The programs in this collection are the engineering realisation of that goal: trained models that convert raw property measurements, market signals, and atmospheric observations into price estimates, sales forecasts, and rainfall probabilities — and visualisations that convert those numerical outputs into insight that a domain expert can trust, question, and act upon.*
