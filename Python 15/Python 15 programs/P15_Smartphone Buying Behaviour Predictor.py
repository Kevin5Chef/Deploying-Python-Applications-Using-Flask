# ============================================================
# AI SMARTPHONE BUYING BEHAVIOR PREDICTOR (FINAL VERSION)
# AD INDEX: 1–10 | BUYING INDEX: 1–100
# WITH FEATURE TRANSPARENCY TABLE
# ============================================================

print("SY-5, Kevin Victor, Roll No.-30")

from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# ============================================================
# 1. DATASET
# ============================================================

np.random.seed(42)
n = 500

ad_index = np.random.uniform(1, 10, n)
features = np.random.uniform(1, 10, n)
marketing = np.random.uniform(1, 10, n)
ai_level = np.random.uniform(1, 10, n)
brand = np.random.uniform(1, 10, n)

# Feature Engineering
influence = (features + ai_level) / 2
market_power = (marketing * brand) / 10

# Target (1–100)
buy_score = (
    0.25*(ad_index/10) +
    0.25*(features/10) +
    0.2*(marketing/10) +
    0.15*(ai_level/10) +
    0.15*(brand/10) +
    0.2*(influence/10) +
    0.1*(market_power/10)
)

buy_score = buy_score * 100
buy_score = np.clip(buy_score, 1, 100)

df = pd.DataFrame({
    "ad": ad_index,
    "features": features,
    "marketing": marketing,
    "ai": ai_level,
    "brand": brand,
    "buy": buy_score
})

# ============================================================
# 2. PIPELINE
# ============================================================

X = df[["ad","features","marketing","ai","brand"]].values
y = df["buy"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
acc = np.mean(np.round(preds) == np.round(y_test)) * 100

print("===== MODEL READY =====")
print("MSE:", round(mse,2))
print("Accuracy:", round(acc,2), "%")

# ============================================================
# 3. TEST SAMPLES
# ============================================================

TEST_SAMPLES = [
    [9,9,8,9,9],
    [4,5,5,6,5],
    [8,8,7,8,7],
    [3,4,3,5,4],
    [6,7,6,7,6]
]

FEATURE_NAMES = [
    "Advertising Index",
    "Features Score",
    "Marketing Strategy",
    "AI Integration",
    "Brand Value"
]

# ============================================================
# 4. FRONTEND
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>AI Smartphone Predictor</title>

<style>
body{
margin:0;
font-family:Arial;
background:linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

.box{
width:750px;
margin:40px auto;
padding:30px;
border-radius:20px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(15px);
box-shadow:0 0 30px rgba(0,255,255,0.2);
}

h1{text-align:center;}

.grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:10px;
}

input{
padding:10px;
border-radius:10px;
border:none;
}

button{
padding:12px;
margin:5px;
border:none;
border-radius:10px;
cursor:pointer;
font-weight:bold;
}

.predict{
background:linear-gradient(90deg,#06b6d4,#3b82f6);
color:white;
}

.reset{
background:red;
color:white;
}

.result{
margin-top:20px;
padding:15px;
background:rgba(255,255,255,0.1);
border-radius:10px;
text-align:center;
font-size:20px;
}

/* TABLE STYLE */
table{
width:100%;
margin-top:20px;
border-collapse:collapse;
background:rgba(255,255,255,0.08);
border-radius:10px;
overflow:hidden;
}

td, th{
padding:10px;
border-bottom:1px solid rgba(255,255,255,0.2);
text-align:left;
}

th{
background:rgba(255,255,255,0.15);
}
</style>

</head>

<body>

<div class="box">
<h1>📱 AI Smartphone Buying Predictor</h1>

<form method="POST">

<div class="grid">
<input name="ad" placeholder="Advertising (1-10)" value="{{vals.ad}}" required>
<input name="features" placeholder="Features (1-10)" value="{{vals.features}}" required>
<input name="marketing" placeholder="Marketing (1-10)" value="{{vals.marketing}}" required>
<input name="ai" placeholder="AI Level (1-10)" value="{{vals.ai}}" required>
<input name="brand" placeholder="Brand Value (1-10)" value="{{vals.brand}}" required>
</div>

<button class="predict" type="submit">Predict</button>
<button class="reset" type="button" onclick="window.location='/'">Reset</button>

</form>

<h3>Quick Samples</h3>
<form method="POST">
<button name="sample" value="0">Sample 1</button>
<button name="sample" value="1">Sample 2</button>
<button name="sample" value="2">Sample 3</button>
<button name="sample" value="3">Sample 4</button>
<button name="sample" value="4">Sample 5</button>
</form>

{% if vals.ad %}
<table>
<tr><th>Feature</th><th>Value</th></tr>
{% for i in range(features|length) %}
<tr>
<td>{{features[i]}}</td>
<td>{{values[i]}}</td>
</tr>
{% endfor %}
</table>
{% endif %}

{% if result %}
<div class="result">
{{result}}
</div>
{% endif %}

</div>

</body>
</html>
"""

# ============================================================
# 5. ROUTE
# ============================================================

@app.route("/", methods=["GET","POST"])
def home():

    result = ""

    vals = {
        "ad":"",
        "features":"",
        "marketing":"",
        "ai":"",
        "brand":""
    }

    values = []

    if request.method == "POST":

        if "sample" in request.form:
            idx = int(request.form["sample"])
            vals["ad"],vals["features"],vals["marketing"],vals["ai"],vals["brand"] = map(str, TEST_SAMPLES[idx])

        try:
            ad = float(vals["ad"] or request.form["ad"])
            features_val = float(vals["features"] or request.form["features"])
            marketing = float(vals["marketing"] or request.form["marketing"])
            ai = float(vals["ai"] or request.form["ai"])
            brand = float(vals["brand"] or request.form["brand"])

            vals = {
                "ad":ad,"features":features_val,"marketing":marketing,"ai":ai,"brand":brand
            }

            values = [ad, features_val, marketing, ai, brand]

            X_input = scaler.transform([[ad,features_val,marketing,ai,brand]])
            pred = model.predict(X_input)[0]

            pred = np.clip(pred, 1, 100)

            result = f"Customer Buying Index: {round(pred,2)} / 100"

        except:
            result = "Invalid Input!"

    return render_template_string(
        HTML,
        result=result,
        vals=vals,
        features=FEATURE_NAMES,
        values=values
    )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)