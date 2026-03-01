<div align="center">

<br/>

```
██████╗██╗  ██╗██╗   ██╗██████╗ ███╗   ██╗    ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗ ██████╗ ██████╗ 
██╔════╝██║  ██║██║   ██║██╔══██╗████╗  ██║    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██║     ███████║██║   ██║██████╔╝██╔██╗ ██║    ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║   ██║██████╔╝
██║     ██╔══██║██║   ██║██╔══██╗██║╚██╗██║    ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║   ██║██╔══██╗
╚██████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║    ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ╚██████╔╝██║  ██║
 ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

### *Predict. Retain. Grow.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)]()

<br/>

> **Acquiring a new customer costs 5–7× more than retaining one.**  
> This system identifies who's about to leave — before they do.

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why It Matters](#-why-it-matters)
- [ML Pipeline](#-ml-pipeline)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [EDA Insights](#-eda-insights)
- [Feature Engineering](#-feature-engineering)
- [Models](#-models)
- [Evaluation](#-evaluation)
- [Getting Started](#-getting-started)
- [Future Roadmap](#-future-roadmap)

---

## 🎯 Overview

**Customer Churn Predictor** is a full end-to-end machine learning system that identifies customers likely to leave a subscription-based service — enabling businesses to take **proactive, data-driven retention actions** before revenue is lost.

This project covers the **complete ML lifecycle**: from raw data to a serialized, deployment-ready model — with a strong emphasis on both **prediction accuracy** and **business interpretability**.

---

## 💡 Why It Matters

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   😤 Customer leaves  →  Revenue lost  →  Costly re-acquisition │
│                                                                 │
│   🤖 Model predicts churn early  →  Team intervenes            │
│   🎁 Targeted offer sent         →  Customer stays             │
│   📈 Revenue protected           →  LTV increases              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Business Problem | ML Solution |
|:---|:---|
| Who will churn next month? | Binary classification model |
| Why are they churning? | Feature importance analysis |
| Who should we prioritize? | Churn probability ranking |
| What's the ROI of intervention? | Threshold-based decision logic |

---

## ⚙️ ML Pipeline

```
  RAW DATA
     │
     ▼
┌────────────┐    ┌──────────────┐    ┌─────────────────┐
│    EDA     │───▶│  Cleaning &  │───▶│    Feature      │
│ & Insights │    │ Preprocessing│    │  Engineering    │
└────────────┘    └──────────────┘    └────────┬────────┘
                                               │
     ┌─────────────────────────────────────────┘
     ▼
┌──────────────────────────────────────────┐
│         Model Training & Comparison      │
│   Logistic Regression | Random Forest    │
│         Gradient Boosting                │
└──────────────────┬───────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │   Evaluation   │
          │ Accuracy · F1  │
          │ Recall · AUC   │
          └───────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  churn_model    │
         │     .pkl  ✅    │
         └─────────────────┘
```

---

## 📁 Project Structure

```
CUSTOMER-CHURN-PREDICTOR/
│
├── 📂 data/
│   ├── 📂 raw/                        # Original, unmodified dataset
│   └── 📂 processed/                  # Cleaned & feature-engineered data
│
├── 📂 notebooks/
│   ├── 📓 EDA.ipynb                   # Exploratory Data Analysis
│   └── 📓 Modeling.ipynb              # Training, evaluation, comparison
│
├── 📂 models/
│   └── 🤖 churn_model.pkl             # Serialized best-performing model
│
├── 📂 src/
│   ├── 🐍 data_processing.py          # Cleaning & preprocessing pipeline
│   ├── 🐍 feature_engineering.py      # Encoding, scaling, transformations
│   ├── 🐍 model_training.py           # Training loop & evaluation
│   └── 🐍 utils.py                    # Shared helper functions
│
├── 📄 requirements.txt                # All dependencies
└── 📄 README.md                       # You are here
```

---

## 📊 Dataset

The dataset contains **real-world telecom customer records** with the following feature categories:

| Category | Features |
|:---|:---|
| 👤 Demographics | Gender, SeniorCitizen, Partner, Dependents |
| 📡 Services | PhoneService, InternetService, StreamingTV, StreamingMovies |
| 📋 Subscription | Contract type, tenure, PaperlessBilling |
| 💳 Payments | PaymentMethod, MonthlyCharges, TotalCharges |
| 🎯 **Target** | **`Churn` → 1 (left) / 0 (stayed)** |

> **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
> **Size:** ~7,000 customer records · 21 features

---

## 🔍 EDA Insights

Key discoveries from exploratory analysis:

```
📉  SHORT TENURE  →  HIGH CHURN RISK
    Customers in their first 12 months churn at nearly 3× the rate
    of long-term customers.

📄  MONTH-TO-MONTH CONTRACTS  →  MOST VOLATILE
    ~42% churn rate vs <5% for two-year contract holders.

💰  HIGH MONTHLY CHARGES  →  ELEVATED RISK
    Customers paying above-average monthly fees show significantly
    higher churn probability.

🌐  FIBER OPTIC USERS  →  SURPRISINGLY HIGH CHURN
    Despite premium service, fiber users churn more — suggesting
    price sensitivity or unmet expectations.
```

---

## 🔧 Feature Engineering

| Step | Description |
|:---|:---|
| 🧹 Missing value handling | Impute or drop based on context |
| 🔠 Categorical encoding | `LabelEncoder` for binary, `OHE` for multi-class |
| ⚖️ Feature scaling | `StandardScaler` for numerical stability |
| ✂️ Redundant feature removal | Drop low-variance & high-correlation columns |
| 🎯 Target encoding | `Churn`: Yes → 1, No → 0 |

---

## 🤖 Models

Three algorithms were trained and benchmarked head-to-head:

| Model | Strength | Use Case |
|:---|:---|:---|
| 📐 **Logistic Regression** | Interpretability, speed | Baseline & explainability |
| 🌲 **Random Forest** | Non-linear patterns, robust | High-dimensional feature interactions |
| 🚀 **Gradient Boosting** | Best accuracy, ensemble power | Final performance push |

---

## 📈 Evaluation

Models are compared across five metrics:

```
  Metric          Why It Matters for Churn
  ─────────────────────────────────────────────────────────
  Accuracy        Overall correctness
  Precision       Are flagged churners actually churning?
  Recall ★        Are we catching all real churners? (key!)
  F1-Score        Balance of precision & recall
  ROC-AUC         Model's ability to rank risk across all thresholds
```

> ⭐ **Recall is the priority metric** — missing a true churner is costlier than a false alarm.

**Best Model Result:**

```
  ╔══════════════════════════════════════╗
  ║  🏆  Gradient Boosting               ║
  ║  ─────────────────────────────────  ║
  ║  Accuracy   →  ~82%                  ║
  ║  Recall     →  ~78%                  ║
  ║  F1-Score   →  ~76%                  ║
  ║  ROC-AUC    →  ~86%                  ║
  ╚══════════════════════════════════════╝
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### 1 · Clone the Repository

```bash
git clone https://github.com/Arshpreet-Singh-2005/CUSTOMER-CHURN-PREDICTOR.git
cd CUSTOMER-CHURN-PREDICTOR
```

### 2 · Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3 · Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 · Add the Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5 · Run the Pipeline

```bash
# Clean and preprocess data
python src/data_processing.py

# Train models and save the best one
python src/model_training.py
```

### 6 · Explore Notebooks (Optional)

```bash
jupyter notebook
```
Open `notebooks/EDA.ipynb` for visual analysis or `notebooks/Modeling.ipynb` for the full training walkthrough.

---

## 🗺️ Future Roadmap

```
 ✅  Phase 1 — Core ML Pipeline          (Complete)
 🔄  Phase 2 — Streamlit Web Interface   (In Progress)
 📋  Phase 3 — REST API via Flask        (Planned)
 🔍  Phase 4 — SHAP Explainability       (Planned)
 ⚡  Phase 5 — Real-time Predictions     (Planned)
 🔁  Phase 6 — CI/CD ML Pipeline         (Planned)
```

---

<div align="center">

---

**Built with 🧠 and ☕ by [Arshpreet Singh](https://github.com/Arshpreet-Singh-2005)**

*If this project helped you, consider giving it a* ⭐

---

</div>
