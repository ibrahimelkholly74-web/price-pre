import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6f0;
}

/* Remove default streamlit padding */
.block-container {
    padding: 2rem 1.5rem 4rem !important;
    max-width: 720px !important;
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-icon {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 24px #7c3aed88);
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Card ── */
.card {
    background: linear-gradient(145deg, #13131f, #1a1a2e);
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px #00000055;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 1.2rem;
}

/* ── Selectbox & Label overrides ── */
label, .stSelectbox label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

div[data-baseweb="select"] > div {
    background-color: #0d0d1a !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:hover {
    border-color: #7c3aed !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px #7c3aed22 !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #13131f !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px !important;
}
li[role="option"]:hover {
    background-color: #1e1e3a !important;
}

/* ── Button ── */
div.stButton > button {
    width: 100%;
    padding: 0.85rem 2rem;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px #7c3aed44;
    margin-top: 0.5rem;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px #7c3aed66;
    background: linear-gradient(135deg, #8b5cf6, #3b82f6);
}
div.stButton > button:active {
    transform: translateY(0px);
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #1a1040, #0f1f3d);
    border: 1px solid #7c3aed55;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 0 40px #7c3aed22;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.result-currency {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #60a5fa;
    margin-top: 0.3rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #3a3a5c;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-bottom: 1rem;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load & Train ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_excel("laptops_training_dataset.xlsx")
    le_brand = LabelEncoder()
    le_condition = LabelEncoder()
    df['Brand'] = le_brand.fit_transform(df['Brand'])
    df['Condition'] = le_condition.fit_transform(df['Condition'])
    X = df.drop('Price', axis=1)
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_brand, le_condition

model, le_brand, le_condition = load_model()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">💻</span>
    <h1>Laptop Price Predictor</h1>
    <p>AI-powered price estimation for the Egyptian market</p>
</div>
""", unsafe_allow_html=True)

# ── Form Card ─────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">📋 Laptop Specifications</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    brand     = st.selectbox("Brand",         le_brand.classes_)
    storage   = st.selectbox("Storage (GB)",  [128, 256, 512, 1024])
    cpu       = st.selectbox("CPU Generation",[4,5,6,7,8,9,10,11])
with col2:
    condition = st.selectbox("Condition",     le_condition.classes_)
    ram       = st.selectbox("RAM (GB)",      [4, 8, 16, 32])
    year      = st.selectbox("Year",          list(range(2018, 2027)))

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict = st.button("⚡ Predict Price")

if predict:
    brand_code     = le_brand.transform([brand])[0]
    condition_code = le_condition.transform([condition])[0]

    new_data = pd.DataFrame({
        'Brand':     [brand_code],
        'RAM':       [ram],
        'Storage':   [storage],
        'CPU_Gen':   [cpu],
        'Year':      [year],
        'Condition': [condition_code]
    })

    price = model.predict(new_data)[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">✨ Estimated Price</div>
        <div class="result-price">{price:,.0f}</div>
        <div class="result-currency">Egyptian Pound · EGP</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Powered by Random Forest · Built with Streamlit</div>', unsafe_allow_html=True)
