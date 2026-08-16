
# Import necessary libraries
import __main__
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Custom transformer class mapping for unpickling
from custom_transformers import SuperKartPreprocessor
__main__.SuperKartPreprocessor = SuperKartPreprocessor

# Initialize Flask app with the standard module name
superkart_api = Flask(__name__)

# Load the trained serialized model pipeline
model = joblib.load("superkart_sales_forecast_model_v1_0.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint to predict sales for a single product-store record
@superkart_api.post('/v1/predict')
def predict_sales():
    data = request.get_json()

    # Extract and map relevant features from input data
    sample = {
        "Product_Weight": float(data["Product_Weight"]),
        "Product_Sugar_Content": data["Product_Sugar_Content"],
        "Product_Allocated_Area": float(data["Product_Allocated_Area"]),
        "Product_MRP": float(data["Product_MRP"]),
        "Store_Size": data["Store_Size"],
        "Store_Location_City_Type": data["Store_Location_City_Type"],
        "Store_Type": data["Store_Type"],
        "Product_Type": data["Product_Type_Category"],
        "Product_Id": data.get("Product_Id_char", "FD") + "001",
        "Store_Establishment_Year": 2026 - int(data["Store_Age_Years"]),
        "Store_Id": data.get("Store_Id", "OUT049"),
    }

    input_data = pd.DataFrame([sample])
    prediction = model.predict(input_data).tolist()[0]

    return jsonify({'Sales': round(prediction, 2)})

# Define an endpoint for bulk CSV upload and inference
@superkart_api.post('/v1/predict-bulk')
def predict_sales_bulk():
    # 1. Validate file presence in multipart form-data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part found. Please upload a CSV file with the form key "file".'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Only .csv files are supported.'}), 400

    try:
        # 2. Read CSV file directly into DataFrame
        input_df = pd.read_csv(file)

        # 3. Handle schema compatibility (supports both raw dataset names and UI input formats)
        if 'Product_Type_Category' in input_df.columns and 'Product_Type' not in input_df.columns:
            input_df['Product_Type'] = input_df['Product_Type_Category']

        if 'Store_Age_Years' in input_df.columns and 'Store_Establishment_Year' not in input_df.columns:
            input_df['Store_Establishment_Year'] = 2026 - input_df['Store_Age_Years'].astype(int)

        if 'Product_Id_char' in input_df.columns and 'Product_Id' not in input_df.columns:
            input_df['Product_Id'] = input_df['Product_Id_char'].astype(str) + "001"

        if 'Store_Id' not in input_df.columns:
            input_df['Store_Id'] = 'OUT049'

        # 4. Perform batch prediction
        predictions = model.predict(input_df).tolist()
        rounded_predictions = [round(p, 2) for p in predictions]

        # 5. Append predictions to DataFrame
        input_df['Predicted_Sales'] = rounded_predictions

        return jsonify({
            'status': 'success',
            'total_records': len(rounded_predictions),
            'predictions': rounded_predictions,
            'data': input_df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process file and generate predictions: {str(e)}'}), 500


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
