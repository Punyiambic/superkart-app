import pandas as pd
import requests
import streamlit as st

# Base API URL configuration
API_BASE_URL = "https://friendly-guacamole-pjj6p47p94xvcrj4p-7860.app.github.dev"

# Application Configuration & Title
st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart Retail Sales Prediction App")
st.markdown("Forecast total product sales revenue using single-record input or bulk CSV batch upload.")

# Single Prediction
st.subheader("Single Record Prediction")
col1, col2 = st.columns(2)

with col1:
    Product_Weight = st.number_input("Product Weight (kg)", min_value=0.0, value=12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, format="%.4f")
    Product_MRP = st.number_input("Product MRP (₹)", min_value=0.0, value=140.0)
    Product_Id_char = st.selectbox("Product ID Code (Prefix)", ["FD", "DR", "NC"])

with col2:
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
    Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=15, step=1)
    Product_Type_Category = st.selectbox(
        "Product Type Category",
        [
            "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods",
            "Dairy", "Canned", "Baking Goods", "Health and Hygiene",
            "Meat", "Soft Drinks", "Breads", "Hard Drinks",
            "Others", "Starchy Foods", "Breakfast", "Seafood"
        ]
    )

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

if st.button("Predict Single Record", type="primary", use_container_width=True):
    predict_url = f"{API_BASE_URL}/v1/predict"
    try:
        response = requests.post(predict_url, json=product_data)
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]
            st.success(f"### Predicted Total Store Sales: ₹{predicted_sales:,.2f}")
        else:
            st.error(f"Error in API request (Status code: {response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Failed to connect to backend server. Ensure the Flask API is running on port 7860.")

st.divider()

# Batch CSV Prediction

st.subheader("Batch Sales Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=["csv"])

if uploaded_file is not None:
    preview_df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded **{len(preview_df)}** records. Preview:")
    st.dataframe(preview_df.head(), use_container_width=True)

    if st.button("Run Batch Prediction", type="secondary", use_container_width=True):
        bulk_url = f"{API_BASE_URL}/v1/predict-bulk"
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")
        }

        with st.spinner("Processing batch predictions..."):
            try:
                response = requests.post(bulk_url, files=files)
                if response.status_code == 200:
                    data = response.json()
                    result_df = pd.DataFrame(data["data"])
                    
                    st.success(f"Successfully processed {data['total_records']} records!")
                    st.dataframe(result_df, use_container_width=True)

                    csv_download = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_download,
                        file_name="superkart_sales_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error(f"API Error (Status code: {response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Failed to connect to backend server. Ensure the Flask API is running on port 7860.")
