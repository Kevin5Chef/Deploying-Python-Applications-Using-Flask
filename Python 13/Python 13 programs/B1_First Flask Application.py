# =========================================================
# BASIC FLASK APPLICATION
# Displays: "Welcome to Flask Application"
# Run locally and open in browser
# =========================================================
print("SY-5, Kevin Victor, Roll No.-30")
from flask import Flask

# Create Flask app object
app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return "Welcome to Flask Application"

# Run local development server
if __name__ == "__main__":
    app.run(debug=True)