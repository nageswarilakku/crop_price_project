from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("crop_price_model.pkl")

# Soil defaults
SOIL_DEFAULTS = {
    "Alluvial soil":     {"N": 75, "P": 40, "K": 40, "TEMP": 26, "HUMIDITY": 80, "RAINFALL": 220, "PH": 6.5},
    "Black soil":        {"N": 80, "P": 45, "K": 50, "TEMP": 28, "HUMIDITY": 75, "RAINFALL": 200, "PH": 6.8},
    "Red soil":          {"N": 70, "P": 35, "K": 30, "TEMP": 27, "HUMIDITY": 78, "RAINFALL": 180, "PH": 6.0},
    "Laterite soil":     {"N": 60, "P": 30, "K": 35, "TEMP": 29, "HUMIDITY": 72, "RAINFALL": 160, "PH": 5.8},
    "Arid soil":         {"N": 40, "P": 20, "K": 15, "TEMP": 35, "HUMIDITY": 50, "RAINFALL": 90,  "PH": 7.2},
    "Forest soil":       {"N": 85, "P": 50, "K": 55, "TEMP": 22, "HUMIDITY": 85, "RAINFALL": 250, "PH": 6.7},
    "Peaty soil":        {"N": 90, "P": 55, "K": 60, "TEMP": 24, "HUMIDITY": 88, "RAINFALL": 280, "PH": 5.5},
    "Loamy soil":        {"N": 78, "P": 42, "K": 46, "TEMP": 25, "HUMIDITY": 76, "RAINFALL": 210, "PH": 6.4},
    "Sandy soil":        {"N": 55, "P": 28, "K": 30, "TEMP": 32, "HUMIDITY": 60, "RAINFALL": 150, "PH": 7.0},
    "Clay soil":         {"N": 65, "P": 38, "K": 42, "TEMP": 23, "HUMIDITY": 82, "RAINFALL": 230, "PH": 6.9},
}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        soil_type = data["SOIL_TYPE"]
        defaults = SOIL_DEFAULTS.get(soil_type, {})

        # Helper function to get value or default
        def get_value(field, default_key):
            val = data.get(field)
            return float(val) if val else defaults.get(default_key, 0)

        # Build input dataframe
        input_df = pd.DataFrame([{
            "STATE": data["STATE"],
            "SOIL_TYPE": soil_type,
            "CROP": data["CROP"],
            "N_SOIL": get_value("N_SOIL", "N"),
            "P_SOIL": get_value("P_SOIL", "P"),
            "K_SOIL": get_value("K_SOIL", "K"),
            "TEMPERATURE": get_value("TEMPERATURE", "TEMP"),
            "HUMIDITY": get_value("HUMIDITY", "HUMIDITY"),
            "ph": get_value("ph", "PH"),
            "RAINFALL": get_value("RAINFALL", "RAINFALL")
        }])

        prediction = model.predict(input_df)[0]
        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
