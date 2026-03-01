import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from feature_engineering import encode_categoricals, scale_features

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*40}")
    print(f"  Model: {name}")
    print(f"{'='*40}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))

    return roc_auc_score(y_test, y_prob)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/processed/churn_cleaned.csv")

    # Feature Engineering
    df = encode_categoricals(df)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        score = evaluate_model(model, X_test_sc, y_test, name)
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\n✅ Best Model: {best_name} (ROC-AUC: {best_score:.4f})")

    # Save best model
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("✅ Model saved to models/churn_model.pkl")
