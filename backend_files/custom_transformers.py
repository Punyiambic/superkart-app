
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SuperKartPreprocessor(BaseEstimator, TransformerMixin):
  """Performs data cleaning, category-level imputations, 
  and feature engineering to prepare data for modeling."""

  def __init__(self, current_year=2026):
    self.current_year = current_year

  def fit(self, X, y=None):
    # Category average MRP
    self.category_mrp_mean_ = X.groupby("Product_Type")["Product_MRP"].mean()

    # Treat 0 allocated area as missing before calculating average
    temp_area = X["Product_Allocated_Area"].replace(0, np.nan)
    self.category_area_mean_ = temp_area.groupby(
        X["Product_Type"]
    ).mean()

    return self

  def transform(self, X):
    X = X.copy()

    # -----------------------
    # Data Cleaning
    # -----------------------

    # Standardize sugar content
    X["Product_Sugar_Content"] = X["Product_Sugar_Content"].replace(
        {"reg": "Regular"}
    )

    # Replace 0 with missing in Allocated Area
    X["Product_Allocated_Area"] = X["Product_Allocated_Area"].replace(
        0, np.nan
    )

    # Fill missing area using Product_Type average
    X["Product_Allocated_Area"] = X["Product_Allocated_Area"].fillna(
        X["Product_Type"].map(self.category_area_mean_)
    )

    # -----------------------
    # Feature Engineering
    # -----------------------

    # Store age
    X["Store_Age"] = self.current_year - X["Store_Establishment_Year"]

    # Product category from first 2 characters of Product_Id
    X["Product_Category_Code"] = X["Product_Id"].str[:2]

    # Price per unit weight
    X["MRP_Per_Weight"] = X["Product_MRP"] / (X["Product_Weight"] + 1e-5)

    # Product price relative to category average
    X["Relative_Category_Price"] = X["Product_MRP"] / X["Product_Type"].map(
        self.category_mrp_mean_
    )

    # Store Type + Store Size interaction (Safely handle NaNs before string concatenation)
    store_size_clean = X["Store_Size"].fillna("Unknown")
    X["Store_Type_Size"] = (
        X["Store_Type"].astype(str) + "_" + store_size_clean.astype(str)
    )

    return X
