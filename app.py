from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("crop_price_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract data from form
        data = request.form

        # Prepare the input DataFrame for prediction
        input_df = pd.DataFrame([{
            "STATE": data["STATE"],
            "SOIL_TYPE": data["SOIL_TYPE"],
            "CROP": data["CROP"],
            "N_SOIL": float(data["N_SOIL"]),
            "P_SOIL": float(data["P_SOIL"]),
            "K_SOIL": float(data["K_SOIL"]),
            "TEMPERATURE": float(data["TEMPERATURE"]),
            "HUMIDITY": float(data["HUMIDITY"]),
            "ph": float(data["ph"]),
            "RAINFALL": float(data["RAINFALL"])
        }])

        # Predict crop price
        prediction = model.predict(input_df)[0]

        # Return result page with prediction
        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
