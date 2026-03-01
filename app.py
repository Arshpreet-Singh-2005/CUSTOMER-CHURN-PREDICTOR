import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.main { background-color: #0a0a0f; }
.block-container { padding: 2rem 3rem; }

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #1a0a2e 50%, #0d1117 100%);
    border: 1px solid #2d1b69;
    border-radius: 20px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,51,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #f472b6, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #9ca3af;
    font-size: 1.1rem;
    margin-top: 0.75rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    color: #a78bfa;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(167,139,250,0.4), transparent);
}

/* ── Cards ── */
.card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.card:hover { border-color: #2d1b69; }

/* ── Result Cards ── */
.result-safe {
    background: linear-gradient(135deg, #022c22, #064e3b);
    border: 1px solid #10b981;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
}
.result-risk {
    background: linear-gradient(135deg, #2d0a0a, #450a0a);
    border: 1px solid #ef4444;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-safe .result-title { color: #10b981; }
.result-risk .result-title { color: #ef4444; }
.result-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-desc { color: #9ca3af; font-size: 0.95rem; margin-top: 0.5rem; }

/* ── Probability Bar ── */
.prob-bar-container {
    background: #1e1e2e;
    border-radius: 50px;
    height: 12px;
    width: 100%;
    margin: 1rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 0.6s ease;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-box {
    flex: 1;
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #a78bfa;
}
.metric-lbl { color: #6b7280; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Sliders & Inputs ── */
.stSlider > div > div > div > div { background: #a78bfa !important; }
.stSelectbox > div > div { background: #111118 !important; border-color: #1e1e2e !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.3s !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 0 35px rgba(124,58,237,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: #1e1e2e !important; }

/* ── Tips ── */
.tip-box {
    background: rgba(167,139,250,0.07);
    border-left: 3px solid #7c3aed;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #c4b5fd;
}
.tip-box strong { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/churn_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem'>🔮</div>
        <div style='font-family: Syne, sans-serif; font-size:1.2rem; font-weight:800; color:#a78bfa;'>CHURN PREDICTOR</div>
        <div style='color:#4b5563; font-size:0.78rem; margin-top:0.3rem;'>ML-Powered Retention Tool</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.markdown('<div class="section-header">📡 Services</div>', unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown('<div class="section-header">📋 Subscription</div>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown('<div class="section-header">💰 Financials</div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges), step=10.0)

    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 PREDICT CHURN RISK")


# ─── MAIN CONTENT ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 Gradient Boosting · ROC-AUC 83.9%</div>
    <h1 class="hero-title">Customer Churn<br/>Prediction System</h1>
    <p class="hero-sub">Fill in the customer profile on the left and click Predict to identify churn risk instantly.</p>
</div>
""", unsafe_allow_html=True)

# ─── MODEL STATUS ───────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model file not found at `models/churn_model.pkl`. Please train the model first by running `python src/model_training.py`")
    st.stop()

# ─── STATS ROW ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="metric-row">
    <div class="metric-box">
        <div class="metric-val">83.9%</div>
        <div class="metric-lbl">ROC-AUC Score</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">82%</div>
        <div class="metric-lbl">Accuracy</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">78%</div>
        <div class="metric-lbl">Recall</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">7043</div>
        <div class="metric-lbl">Training Records</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── PREDICTION ─────────────────────────────────────────────────────────────
if predict_btn:
    # Build input
    def encode(val, mapping): return mapping.get(val, 0)

    binary = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    three_way = {"No": 0, "No phone service": 0, "No internet service": 0, "Yes": 1}

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    multiline_map = {"No": 0, "No phone service": 1, "Yes": 2}
    three_service_map = {"No": 0, "No internet service": 1, "Yes": 2}

    input_data = np.array([[
        encode(gender, binary),
        encode(senior, binary),
        encode(partner, binary),
        encode(dependents, binary),
        tenure,
        encode(phone_service, binary),
        encode(multiple_lines, multiline_map),
        encode(internet_service, internet_map),
        encode(online_security, three_service_map),
        encode(online_backup, three_service_map),
        encode(device_protection, three_service_map),
        encode(tech_support, three_service_map),
        encode(streaming_tv, three_service_map),
        encode(streaming_movies, three_service_map),
        encode(contract, contract_map),
        encode(paperless, binary),
        encode(payment, payment_map),
        monthly_charges,
        total_charges
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    prob_pct = round(probability * 100, 1)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-risk">
                <div class="result-emoji">⚠️</div>
                <div class="result-title">HIGH CHURN RISK</div>
                <div style="font-size:3rem; font-weight:800; color:#ef4444; font-family:Syne,sans-serif;">{prob_pct}%</div>
                <div class="result-desc">This customer is likely to leave. Immediate retention action recommended.</div>
                <div class="prob-bar-container" style="margin-top:1.5rem;">
                    <div class="prob-bar-fill" style="width:{prob_pct}%; background: linear-gradient(90deg, #ef4444, #dc2626);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-emoji">✅</div>
                <div class="result-title">LOW CHURN RISK</div>
                <div style="font-size:3rem; font-weight:800; color:#10b981; font-family:Syne,sans-serif;">{prob_pct}%</div>
                <div class="result-desc">This customer is likely to stay. Keep up the good engagement!</div>
                <div class="prob-bar-container" style="margin-top:1.5rem;">
                    <div class="prob-bar-fill" style="width:{prob_pct}%; background: linear-gradient(90deg, #10b981, #059669);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">📋 Customer Summary</div>', unsafe_allow_html=True)
        summary_data = {
            "Contract": contract,
            "Tenure": f"{tenure} months",
            "Monthly Charges": f"${monthly_charges}",
            "Internet Service": internet_service,
            "Tech Support": tech_support,
            "Senior Citizen": senior,
        }
        for k, v in summary_data.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.6rem 0; border-bottom:1px solid #1e1e2e; font-size:0.9rem;">
                <span style="color:#6b7280;">{k}</span>
                <span style="color:#e8e8f0; font-weight:500;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Retention Tips ──
    st.markdown('<div class="section-header">💡 Retention Recommendations</div>', unsafe_allow_html=True)

    tips = []
    if contract == "Month-to-month":
        tips.append(("<strong>Upgrade Contract</strong>", "Offer a discount to switch to a 1 or 2-year contract. Month-to-month users churn 3× more."))
    if tenure < 12:
        tips.append(("<strong>Early Loyalty Program</strong>", "Customer is in the high-risk first year. Enroll them in an onboarding rewards program."))
    if monthly_charges > 70:
        tips.append(("<strong>Price Sensitivity Check</strong>", "High monthly charges detected. Consider a personalized discount or bundle offer."))
    if tech_support == "No" and internet_service != "No":
        tips.append(("<strong>Offer Tech Support</strong>", "Customers without tech support churn more. Offer a free 3-month trial."))
    if online_security == "No" and internet_service != "No":
        tips.append(("<strong>Add Online Security</strong>", "Bundling security services increases perceived value and reduces churn likelihood."))
    if not tips:
        tips.append(("<strong>Maintain Engagement</strong>", "Customer profile looks stable. Continue regular check-ins and loyalty rewards."))

    for title, desc in tips:
        st.markdown(f'<div class="tip-box">{title} — {desc}</div>', unsafe_allow_html=True)

else:
    # ── Default state ──
    st.markdown("""
    <div class="card" style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🔮</div>
        <div style="font-family:Syne,sans-serif; font-size:1.5rem; font-weight:700; color:#a78bfa;">Ready to Predict</div>
        <div style="color:#6b7280; margin-top:0.5rem; font-size:0.95rem;">
            Fill in the customer details in the sidebar<br/>and click <strong style="color:#a78bfa;">PREDICT CHURN RISK</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Key Churn Drivers</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    drivers = [
        ("📄", "Contract Type", "Month-to-month customers churn at 42% vs <5% for 2-year contracts"),
        ("📅", "Short Tenure", "Customers in their first 12 months are 3× more likely to churn"),
        ("💸", "High Charges", "Above-average monthly bills strongly correlate with churn risk"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], drivers):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-family:Syne,sans-serif; font-weight:700; color:#a78bfa; margin:0.5rem 0;">{title}</div>
                <div style="color:#6b7280; font-size:0.85rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr/>
<div style="text-align:center; color:#374151; font-size:0.8rem; padding:1rem 0;">
    Built by <strong style="color:#a78bfa;">Arshpreet Singh</strong> ·
    Gradient Boosting Model · Telco Customer Churn Dataset
</div>
""", unsafe_allow_html=True)
