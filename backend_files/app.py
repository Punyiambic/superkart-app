import __main__
# Import necessary libraries
import joblib  # For loading the serialized model
import numpy as np
import pandas as pd  # For data manipulation
from custom_transformers import SuperKartPreprocessor
from flask import Flask, jsonify, request  # For creating the Flask API

# Inject the class into __main__ so joblib/pickle can find it
__main__.SuperKartPreprocessor = SuperKartPreprocessor

# Initialize Flask app with the standard module name
superkart_api = Flask(__name__)

# Load the trained serialized model pipeline
model = joblib.load("superkart_sales_forecast_model_v1_0.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint to predict sales for a product-store record
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the incoming POST request
    data = request.get_json()

    sample = {
        "Product_Weight": float(data["Product_Weight"]),
        "Product_Sugar_Content": data["Product_Sugar_Content"],
        "Product_Allocated_Area": float(data["Product_Allocated_Area"]),
        "Product_MRP": float(data["Product_MRP"]),
        "Store_Size": data["Store_Size"],
        "Store_Location_City_Type": data["Store_Location_City_Type"],
        "Store_Type": data["Store_Type"],
        # Rename to the exact column names required by your pipeline:
        "Product_Type": data["Product_Type_Category"],
        "Product_Id": data.get("Product_Id_char", "FD")
        + "001",  # Ensures .str[:2] works
        "Store_Establishment_Year": 2026 - int(data["Store_Age_Years"]),
        "Store_Id": data.get(
            "Store_Id", "OUT049"
        ),  # Required by nominal ColumnTransformer
    }
    
    # Convert the extracted dictionary into a single-row DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction using the loaded model pipeline
    prediction = model.predict(input_data).tolist()[0]

    # Return prediction as JSON response
    return jsonify({'Sales': prediction})

# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
