# ==========================================================
# BMI CALCULATOR API + MODERN FRONTEND (FLASK)
# ----------------------------------------------------------
# Features:
# ✔ Accepts Height (cm) and Weight (kg)
# ✔ Calculates BMI
# ✔ WHO-style BMI categories
# ✔ Color-coded result cards
# ✔ Glassmorphism UI
# ✔ Responsive design
# ✔ Smooth animations
# ✔ Popup-style result overlay
#
# RUN:
# python app.py
#
# OPEN:
# http://127.0.0.1:5000/
# ==========================================================

from flask import Flask, request, render_template_string
print("SY-5, Kevin Victor, Roll No.-30")
app = Flask(__name__)

# ==========================================================
# BMI LOGIC
# Common public BMI adult ranges:
# <18.5 Underweight
# 18.5-24.9 Normal
# 25-29.9 Overweight
# >=30 Obesity
# ==========================================================
def bmi_category(bmi):

    if bmi < 18.5:
        return "Underweight", "#00c2ff", "Consider nutrition-focused weight gain with professional guidance."

    elif bmi < 25:
        return "Normal Weight", "#00e676", "Healthy range. Maintain activity, sleep, and balanced nutrition."

    elif bmi < 30:
        return "Overweight", "#ffb300", "Gradual lifestyle adjustments can improve long-term health."

    else:
        return "Obesity", "#ff5252", "Consider structured medical and fitness support."

# ==========================================================
# HTML TEMPLATE
# ==========================================================
PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BMI Calculator</title>

<style>
*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

body{
    min-height:100vh;
    display:flex;
    justify-content:center;
    align-items:center;
    padding:20px;
    font-family:Arial, Helvetica, sans-serif;
    background:
        radial-gradient(circle at top left,#1e3c72,transparent 35%),
        radial-gradient(circle at bottom right,#2a5298,transparent 35%),
        linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    overflow:hidden;
}

/* floating HUD glow */
body::before,
body::after{
    content:"";
    position:fixed;
    width:280px;
    height:280px;
    border-radius:50%;
    filter:blur(70px);
    opacity:.35;
    animation:floatGlow 8s infinite ease-in-out;
}

body::before{
    background:#00e5ff;
    top:8%;
    left:8%;
}

body::after{
    background:#7c4dff;
    bottom:8%;
    right:8%;
    animation-delay:3s;
}

@keyframes floatGlow{
    0%,100%{transform:translateY(0px) scale(1);}
    50%{transform:translateY(-25px) scale(1.08);}
}

.panel{
    width:100%;
    max-width:430px;
    padding:32px;
    border-radius:24px;
    background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.22);
    backdrop-filter:blur(16px);
    -webkit-backdrop-filter:blur(16px);
    box-shadow:0 20px 50px rgba(0,0,0,.35);
    animation:slideUp .8s ease;
}

@keyframes slideUp{
    from{opacity:0;transform:translateY(35px);}
    to{opacity:1;transform:translateY(0);}
}

h1{
    color:white;
    text-align:center;
    font-size:30px;
    margin-bottom:8px;
}

.sub{
    text-align:center;
    color:rgba(255,255,255,.78);
    margin-bottom:24px;
    font-size:14px;
}

label{
    display:block;
    margin-top:16px;
    margin-bottom:8px;
    color:white;
    font-weight:bold;
}

input{
    width:100%;
    padding:14px;
    border:none;
    border-radius:14px;
    outline:none;
    background:rgba(255,255,255,0.18);
    color:white;
    font-size:16px;
    transition:.25s;
}

input:focus{
    transform:scale(1.02);
    background:rgba(255,255,255,0.24);
}

input::placeholder{
    color:rgba(255,255,255,.7);
}

button{
    width:100%;
    margin-top:24px;
    padding:15px;
    border:none;
    border-radius:14px;
    font-size:17px;
    font-weight:bold;
    cursor:pointer;
    color:white;
    background:linear-gradient(135deg,#00c6ff,#0072ff);
    transition:.25s;
}

button:hover{
    transform:translateY(-2px);
    box-shadow:0 12px 22px rgba(0,114,255,.35);
}

.popup{
    margin-top:22px;
    border-radius:18px;
    padding:20px;
    background:rgba(255,255,255,0.14);
    border:1px solid rgba(255,255,255,0.18);
    animation:fadeIn .7s ease;
}

@keyframes fadeIn{
    from{opacity:0;transform:scale(.92);}
    to{opacity:1;transform:scale(1);}
}

.metric{
    color:white;
    font-size:28px;
    font-weight:bold;
    text-align:center;
}

.category{
    text-align:center;
    font-size:20px;
    font-weight:bold;
    margin-top:8px;
}

.tip{
    color:rgba(255,255,255,.88);
    margin-top:10px;
    text-align:center;
    line-height:1.45;
    font-size:14px;
}

.footer{
    margin-top:18px;
    text-align:center;
    color:rgba(255,255,255,.65);
    font-size:12px;
}

@media(max-width:520px){
    .panel{
        padding:24px;
    }
    h1{
        font-size:25px;
    }
}
</style>
</head>

<body>

<div class="panel">

    <h1>BMI Calculator</h1>
    <div class="sub">Enter your height and weight</div>

    <form method="POST">

        <label>Height (cm)</label>
        <input type="number" step="0.1" name="height" placeholder="e.g. 170" required>

        <label>Weight (kg)</label>
        <input type="number" step="0.1" name="weight" placeholder="e.g. 68" required>

        <button type="submit">Calculate BMI</button>

    </form>

    {% if bmi %}
    <div class="popup">
        <div class="metric">BMI {{ bmi }}</div>
        <div class="category" style="color:{{ color }};">{{ category }}</div>
        <div class="tip">{{ message }}</div>
    </div>
    {% endif %}

    <div class="footer">General informational ranges for adults.</div>

</div>

</body>
</html>
"""

# ==========================================================
# MAIN ROUTE
# ==========================================================
@app.route("/", methods=["GET", "POST"])
def home():

    bmi = None
    category = ""
    color = ""
    message = ""

    if request.method == "POST":
        try:
            height_cm = float(request.form["height"])
            weight_kg = float(request.form["weight"])

            if height_cm <= 0 or weight_kg <= 0:
                raise ValueError

            height_m = height_cm / 100.0
            bmi_val = weight_kg / (height_m ** 2)

            category, color, message = bmi_category(bmi_val)
            bmi = round(bmi_val, 2)

        except:
            bmi = "Invalid input"
            category = ""
            color = "#ffffff"
            message = "Please enter valid positive numbers."

    return render_template_string(
        PAGE,
        bmi=bmi,
        category=category,
        color=color,
        message=message
    )

# ==========================================================
# START APP
# ==========================================================
if __name__ == "__main__":
    print("BMI Calculator running...")
    print("Open: http://127.0.0.1:5000/")
    app.run(debug=True)