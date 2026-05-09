# =========================================================
# FULL FLASK REST API PROGRAM
# 1. Start Flask server first
# 2. Then ask user for inputs on PyCharm terminal
# 3. Display sum in JSON format on localhost
#
# HOW IT WORKS:
# - Flask server starts
# - A background thread asks terminal user for a and b
# - Browser route "/" returns latest values + sum in JSON
# =========================================================

from flask import Flask, jsonify
import threading
import time
print("SY-5, Kevin Victor, Roll No.-30")
# ---------------------------------------------------------
# Create Flask app
# ---------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------
# Global variables
# These store latest terminal inputs
# ---------------------------------------------------------
a = None
b = None
result = None


# ---------------------------------------------------------
# TERMINAL INPUT FUNCTION
# Runs AFTER Flask starts
# Keeps asking user for numbers in PyCharm terminal
# ---------------------------------------------------------
def take_inputs():

    global a, b, result

    while True:
        try:
            print("\nEnter values in PyCharm terminal")

            a = float(input("Enter first number (a): "))
            b = float(input("Enter second number (b): "))

            result = a + b

            print("Values stored successfully.")
            print("Open browser: http://127.0.0.1:5000/\n")

        except ValueError:
            print("Invalid input. Please enter numbers only.")


# ---------------------------------------------------------
# HOME ROUTE
# Shows latest values in JSON format
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def show_sum():

    if a is None or b is None:
        return jsonify({
            "message": "Flask started. Please enter values in PyCharm terminal."
        })

    return jsonify({
        "first_number": a,
        "second_number": b,
        "sum": result
    })


# ---------------------------------------------------------
# MAIN PROGRAM
# Start Flask first, then start terminal input thread
# ---------------------------------------------------------
if __name__ == "__main__":

    # Start terminal input in background thread
    input_thread = threading.Thread(target=take_inputs, daemon=True)
    input_thread.start()

    # Small delay so server message appears first
    time.sleep(1)

    # Start Flask server
    app.run(debug=True, use_reloader=False)