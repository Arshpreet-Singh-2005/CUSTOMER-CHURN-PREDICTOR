import pandas as pd
import numpy as np
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Fix TotalCharges column (has spaces instead of NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Drop customerID (not useful for prediction)
    df.drop(columns=['customerID'], inplace=True)

    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def save_processed(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    out_path = "data/processed/churn_cleaned.csv"

    df = load_data(raw_path)
    df = clean_data(df)
    save_processed(df, out_path)
