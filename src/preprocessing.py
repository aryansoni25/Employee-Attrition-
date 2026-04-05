import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 🔹 Base paths (IMPORTANT for Docker + API)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "employee.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create models directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)


# 🔹 Load data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)


# 🔹 Clean data
def clean_data(df):
    drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Convert target
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    return df


# 🔹 Encode categorical features
def encode_data(df):
    le = LabelEncoder()

    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df


# 🔹 Feature selection
def feature_selection(df, threshold=0.05):
    corr = df.corr()['Attrition'].abs().sort_values(ascending=False)
    selected = corr[corr > threshold].index
    return df[selected]


# 🔹 Train-test split
def split_data(df):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    return train_test_split(X, y, test_size=0.2, random_state=42)


# 🔹 Scaling
def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return X_train, X_test


# 🔹 Full pipeline
def preprocess_pipeline():
    df = load_data()
    df = clean_data(df)
    df = encode_data(df)
    df = feature_selection(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # ✅ Save feature names (for Streamlit UI)
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    # ✅ Save input size (for model + UI)
    joblib.dump(X_train.shape[1], os.path.join(MODEL_DIR, "input_size.pkl"))

    # Scaling
    X_train, X_test = scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test