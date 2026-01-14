import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class Preprocessing:
    def __init__(self):
        pass
        
    def load_data(self, data_path):
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"The file {data_path} does not exist.")
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e
    
    def clean_data(self, data):
        try:
            # changing the dtype of 'TotalCharges' to numeric, coercing errors to NaN
            data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
            data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})
            
            # removing missing values
            data = data.dropna()
            
            # drop ID column if exists
            if 'customerID' in data.columns:
                data = data.drop('customerID', axis=1)
            return data
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            raise e
    
    def build_processor(self, X):
        try:
            # identify categorical and numerical columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            numerical_cols =X.select_dtypes(include=['int64', 'float64']).columns
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                    ('num', 'passthrough', numerical_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            print(f"Error during building preprocessor: {e}")
            raise e
    
    def save_clean_data(self, data, output_path):
        try:
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            data.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error during saving clean data: {e}")
            raise e