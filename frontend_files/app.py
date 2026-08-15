import requests
import streamlit as st

# Application Title
st.title("SuperKart Retail Sales Prediction App")

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight (kg)", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox(
    "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
)
Product_Allocated_Area = st.number_input(
    "Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, format="%.4f"
)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=140.0)

Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox(
    "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
)
Store_Type = st.selectbox(
    "Store Type",
    ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"],
)

Product_Id_char = st.selectbox(
    "Product ID Code (Prefix)", ["FD", "DR", "NC"]
)
Store_Age_Years = st.number_input(
    "Store Age (Years)", min_value=0, max_value=100, value=15, step=1
)
Product_Type_Category = st.selectbox(
    "Product Type Category",
    [
        "Fruits and Vegetables",
        "Snack Foods",
        "Household",
        "Frozen Foods",
        "Dairy",
        "Canned",
        "Baking Goods",
        "Health and Hygiene",
        "Meat",
        "Soft Drinks",
        "Breads",
        "Hard Drinks",
        "Others",
        "Starchy Foods",
        "Breakfast",
        "Seafood",
    ],
)

# Prepare dictionary payload matching backend API schema
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category,
}

# Prediction button and API request
if st.button("Predict", type="primary"):
    # If deploying to Hugging Face Spaces, replace URL with your space endpoint:
    # e.g., https://username-space-name.hf.space/v1/predict
    api_url = "http://localhost:7860/v1/predict"

    try:
        response = requests.post(api_url, json=product_data)
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]
            st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:,.2f}")
        else:
            st.error(f"Error in API request: Status code {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend server. Make sure the Flask/Gunicorn API is running.")
