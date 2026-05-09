# ==========================================================
# FLASK TEMPERATURE BAR CHART APP
# ----------------------------------------------------------
# Features:
# ✔ Enter max/min temperatures for last 5 days
# ✔ Accepts integers and decimals
# ✔ Generates styled bar chart on webpage
# ✔ Input values remain after clicking OK
# ✔ Separate Reset button
# ✔ Glassmorphism UI
# ✔ Responsive layout
# ✔ Gradient bars + grid chart
#
# INSTALL:
# pip install flask matplotlib
#
# RUN:
# python app.py
#
# OPEN:
# http://127.0.0.1:5000/
# ==========================================================

from flask import Flask, request, render_template_string
import matplotlib
matplotlib.use("Agg")   # server-side rendering
import matplotlib.pyplot as plt
import io
import base64
print("SY-5, Kevin Victor, Roll No.-30")
app = Flask(__name__)

# ==========================================================
# CREATE CHART
# ==========================================================
def generate_chart(max_vals, min_vals):

    days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    x = range(len(days))
    width = 0.36

    # Gradient-like colors
    max_colors = ["#ff6b6b", "#ff7b54", "#ff8c42", "#ff9f1c", "#ffb703"]
    min_colors = ["#4dabf7", "#339af0", "#228be6", "#1c7ed6", "#1971c2"]

    ax.bar(
        [i - width/2 for i in x],
        max_vals,
        width=width,
        color=max_colors,
        label="Max Temp (°C)",
        edgecolor="white",
        linewidth=1.2
    )

    ax.bar(
        [i + width/2 for i in x],
        min_vals,
        width=width,
        color=min_colors,
        label="Min Temp (°C)",
        edgecolor="white",
        linewidth=1.2
    )

    ax.set_title("Last 5 Days Temperature Analysis", fontsize=16, weight="bold")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(days)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    # Cleaner frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save to memory
    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    img.seek(0)

    return base64.b64encode(img.getvalue()).decode()


# ==========================================================
# HTML TEMPLATE
# ==========================================================
PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Temperature Bar Chart</title>

<style>
*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

body{
    min-height:100vh;
    font-family:Arial, Helvetica, sans-serif;
    display:flex;
    justify-content:center;
    align-items:flex-start;
    padding:30px;
    background:
        radial-gradient(circle at top left,#00e5ff22,transparent 30%),
        radial-gradient(circle at bottom right,#7c4dff22,transparent 30%),
        linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.panel{
    width:100%;
    max-width:980px;
    background:rgba(255,255,255,.10);
    border:1px solid rgba(255,255,255,.18);
    backdrop-filter:blur(16px);
    border-radius:24px;
    padding:28px;
    box-shadow:0 18px 45px rgba(0,0,0,.35);
    animation:rise .7s ease;
}

@keyframes rise{
    from{opacity:0; transform:translateY(25px);}
    to{opacity:1; transform:translateY(0);}
}

h1{
    color:white;
    text-align:center;
    margin-bottom:8px;
}

.sub{
    text-align:center;
    color:#dbe7ff;
    margin-bottom:22px;
}

.grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:14px;
}

.card{
    background:rgba(255,255,255,.08);
    border-radius:16px;
    padding:14px;
}

label{
    display:block;
    color:white;
    font-size:14px;
    margin-bottom:6px;
}

input{
    width:100%;
    padding:10px;
    border:none;
    border-radius:10px;
    outline:none;
    font-size:15px;
    background:rgba(255,255,255,.18);
    color:white;
}

input::placeholder{
    color:#e3e3e3;
}

.buttons{
    margin-top:20px;
    display:flex;
    gap:12px;
    flex-wrap:wrap;
}

button, .reset{
    padding:12px 22px;
    border:none;
    border-radius:12px;
    cursor:pointer;
    font-size:15px;
    font-weight:bold;
    text-decoration:none;
    color:white;
}

button{
    background:linear-gradient(135deg,#00c6ff,#0072ff);
}

button:hover{
    transform:translateY(-2px);
}

.reset{
    background:linear-gradient(135deg,#ff6b6b,#d6336c);
    display:inline-flex;
    align-items:center;
}

.chart-box{
    margin-top:28px;
    background:rgba(255,255,255,.08);
    border-radius:18px;
    padding:18px;
    animation:fade .6s ease;
}

@keyframes fade{
    from{opacity:0;}
    to{opacity:1;}
}

img{
    width:100%;
    border-radius:12px;
}

.error{
    margin-top:16px;
    color:#ffdede;
    background:#b00020aa;
    padding:12px;
    border-radius:10px;
}

@media(max-width:700px){
    .panel{padding:18px;}
}
</style>
</head>

<body>

<div class="panel">

<h1>Temperature Bar Chart</h1>
<div class="sub">Enter maximum and minimum temperatures for the last 5 days</div>

<form method="POST">

<div class="grid">

{% for i in range(5) %}
<div class="card">
<label>Day {{i+1}} Max (°C)</label>
<input type="number" step="any" name="max{{i}}" value="{{ values['max' ~ i] }}">
<label style="margin-top:10px;">Day {{i+1}} Min (°C)</label>
<input type="number" step="any" name="min{{i}}" value="{{ values['min' ~ i] }}">
</div>
{% endfor %}

</div>

<div class="buttons">
<button type="submit">OK</button>
<a class="reset" href="/">Reset</a>
</div>

</form>

{% if error %}
<div class="error">{{ error }}</div>
{% endif %}

{% if chart %}
<div class="chart-box">
<img src="data:image/png;base64,{{ chart }}">
</div>
{% endif %}

</div>

</body>
</html>
"""

# ==========================================================
# ROUTE
# ==========================================================
@app.route("/", methods=["GET", "POST"])
def home():

    chart = None
    error = ""

    # preserve values
    values = {}
    for i in range(5):
        values[f"max{i}"] = ""
        values[f"min{i}"] = ""

    if request.method == "POST":
        try:
            max_vals = []
            min_vals = []

            for i in range(5):
                max_key = f"max{i}"
                min_key = f"min{i}"

                values[max_key] = request.form.get(max_key, "")
                values[min_key] = request.form.get(min_key, "")

                mx = float(values[max_key])
                mn = float(values[min_key])

                if mn > mx:
                    raise ValueError(f"Day {i+1}: Min cannot exceed Max.")

                max_vals.append(mx)
                min_vals.append(mn)

            chart = generate_chart(max_vals, min_vals)

        except Exception as e:
            error = str(e)

    return render_template_string(
        PAGE,
        chart=chart,
        values=values,
        error=error
    )


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("Temperature Chart App Running...")
    print("Open: http://127.0.0.1:5000/")
    app.run(debug=True)