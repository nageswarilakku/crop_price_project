import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Load the dataset
df = pd.read_csv("crop_data.csv")

# Step 2: Define features and target
X = df.drop("CROP_PRICE", axis=1)
y = df["CROP_PRICE"]

# Step 3: Identify categorical and numerical columns
categorical_cols = ["STATE", "SOIL_TYPE", "CROP"]
numerical_cols = X.drop(columns=categorical_cols).columns.tolist()

# Step 4: Create a preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

# Step 5: Combine preprocessor with model
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model_pipeline.fit(X_train, y_train)

# Step 8: Evaluate model performance
y_pred = model_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 9: Save the trained model
joblib.dump(model_pipeline, "crop_price_model.pkl")
print("✅ Model saved as 'crop_price_model.pkl'")

# Step 10: Load and use the model (example)
loaded_model = joblib.load("crop_price_model.pkl")

# Example prediction (optional test data)
sample_data = pd.DataFrame([{
    "STATE": "Karnataka",
    "SOIL_TYPE": "Loamy soil",
    "CROP": "Rice",
    "N_SOIL": 80,
    "P_SOIL": 40,
    "K_SOIL": 45,
    "TEMPERATURE": 25.0,
    "HUMIDITY": 75.0,
    "ph": 6.5,
    "RAINFALL": 300.0
}])
predicted_price = loaded_model.predict(sample_data)[0]
print(f"Predicted Price: ₹{round(predicted_price, 2)}")
