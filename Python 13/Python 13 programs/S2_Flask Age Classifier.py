# ==========================================================
# AGE CLASSIFICATION SERVICE WITH MINIMAL UI (FLASK)
# ----------------------------------------------------------
# Features:
# 1. Minimal frontend UI
# 2. Enter age in textbox
# 3. Click OK button
# 4. Output shown instantly
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
# AGE CLASSIFICATION LOGIC
# ==========================================================
def classify_age(age):

    if age < 0:
        return "Invalid Age"

    elif age < 12:
        return "Child"

    elif age > 65:
        return "Senior Citizen"

    elif age > 18:
        return "Adult"

    else:
        return "Teenager / Minor"


# ==========================================================
# HTML FRONTEND
# ==========================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Age Classifier</title>

    <style>
        body{
            font-family: Arial;
            background:#f5f5f5;
            text-align:center;
            margin-top:120px;
        }

        .box{
            background:white;
            width:320px;
            margin:auto;
            padding:30px;
            border-radius:10px;
            box-shadow:0px 0px 10px rgba(0,0,0,0.15);
        }

        input{
            width:220px;
            padding:10px;
            font-size:16px;
            margin-top:10px;
        }

        button{
            margin-top:15px;
            padding:10px 25px;
            font-size:16px;
            cursor:pointer;
            background:#007bff;
            color:white;
            border:none;
            border-radius:5px;
        }

        button:hover{
            background:#0056b3;
        }

        .result{
            margin-top:20px;
            font-size:20px;
            color:green;
            font-weight:bold;
        }
    </style>
</head>

<body>

<div class="box">
    <h2>Age Classification</h2>

    <form method="POST">
        <input type="number" name="age" placeholder="Enter age" required>
        <br>
        <button type="submit">OK</button>
    </form>

    {% if result %}
        <div class="result">{{ result }}</div>
    {% endif %}
</div>

</body>
</html>
"""


# ==========================================================
# MAIN ROUTE
# ==========================================================
@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        try:
            age = int(request.form["age"])
            result = classify_age(age)

        except:
            result = "Invalid Input"

    return render_template_string(HTML_PAGE, result=result)


# ==========================================================
# START SERVER
# ==========================================================
if __name__ == "__main__":
    print("Age Classifier Running...")
    print("Open browser: http://127.0.0.1:5000/")
    app.run(debug=True)