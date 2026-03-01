import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_categoricals(df):
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
