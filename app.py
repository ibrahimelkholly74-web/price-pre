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

# ── Language Strings ───────────────────────────────────────────────────────────
LANG = {
    "en": {
        "title": "Laptop Price Predictor",
        "subtitle": "AI-powered price estimation for the Egyptian market",
        "section": "📋 Laptop Specifications",
        "brand": "Brand",
        "storage": "Storage (GB)",
        "cpu": "CPU Generation",
        "condition": "Condition",
        "ram": "RAM (GB)",
        "year": "Year",
        "button": "⚡ Predict Price",
        "est_label": "✨ Estimated Price Range",
        "low": "Low",
        "mid": "Mid",
        "high": "High",
        "currency_label": "💱 Currency",
        "footer": "Powered by Random Forest · Built with Streamlit",
        "dir": "ltr",
    },
    "ar": {
        "title": "توقع سعر اللاب توب",
        "subtitle": "تقدير الأسعار بالذكاء الاصطناعي للسوق المصري",
        "section": "📋 مواصفات اللاب توب",
        "brand": "الماركة",
        "storage": "التخزين (جيجا)",
        "cpu": "جيل المعالج",
        "condition": "الحالة",
        "ram": "الرام (جيجا)",
        "year": "سنة الصنع",
        "button": "⚡ توقع السعر",
        "est_label": "✨ نطاق السعر المتوقع",
        "low": "الأدنى",
        "mid": "المتوسط",
        "high": "الأعلى",
        "currency_label": "💱 العملة",
        "footer": "يعمل بـ Random Forest · مبني بـ Streamlit",
        "dir": "rtl",
    }
}

# ── Currency rates (relative to EGP) ──────────────────────────────────────────
CURRENCIES = {
    "🇪🇬 EGP": 1.0,
    "🇺🇸 USD": 0.021,
    "🇪🇺 EUR": 0.019,
    "🇸🇦 SAR": 0.079,
    "🇦🇪 AED": 0.077,
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=Cairo:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e8e6f0;
}

.block-container {
    padding: 2rem 1.5rem 4rem !important;
    max-width: 720px !important;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 24px #7c3aed88);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 300;
}

/* Card */
.card {
    background: linear-gradient(145deg, #13131f, #1a1a2e);
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px #00000055;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 1.2rem;
}

/* Labels */
label, .stSelectbox label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #0d0d1a !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #e8e6f0 !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:hover { border-color: #7c3aed !important; }
div[data-baseweb="select"] > div:focus-within {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px #7c3aed22 !important;
}
ul[role="listbox"] {
    background-color: #13131f !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px !important;
}
li[role="option"]:hover { background-color: #1e1e3a !important; }

/* Button */
div.stButton > button {
    width: 100%;
    padding: 0.85rem 2rem;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #fff;
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

/* Range result boxes */
.range-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-top: 1.2rem;
}
.range-box {
    border-radius: 16px;
    padding: 1.2rem 0.8rem;
    text-align: center;
    animation: fadeUp 0.4s ease;
}
.range-box.low  { background: #0f1f2a; border: 1px solid #0ea5e955; }
.range-box.mid  { background: #1a1040; border: 1px solid #7c3aed88; box-shadow: 0 0 30px #7c3aed22; }
.range-box.high { background: #1f0f2a; border: 1px solid #f472b655; }
.range-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.low  .range-label { color: #38bdf8; }
.mid  .range-label { color: #a78bfa; }
.high .range-label { color: #f472b6; }
.range-price {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #e8e6f0;
}
.range-currency {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 0.2rem;
}
.result-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 0.2rem;
    text-align: center;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* RTL support */
.rtl { direction: rtl; text-align: right; font-family: 'Cairo', sans-serif !important; }
.rtl label, .rtl .card-title, .rtl .result-title { font-family: 'Cairo', sans-serif !important; letter-spacing: 0 !important; }

/* Lang toggle */
.stRadio > div { flex-direction: row !important; gap: 0.5rem; }
.stRadio label { font-size: 0.85rem !important; color: #a78bfa !important; }

/* Footer */
.footer {
    text-align: center;
    color: #3a3a5c;
    font-size: 0.78rem;
    margin-top: 3rem;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load & Train ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_excel("laptops_training_dataset.xlsx")
    le_brand = LabelEncoder()
    le_condition = LabelEncoder()
    df['Brand']     = le_brand.fit_transform(df['Brand'])
    df['Condition'] = le_condition.fit_transform(df['Condition'])
    X = df.drop('Price', axis=1)
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_brand, le_condition

model, le_brand, le_condition = load_model()

# ── Language Toggle ────────────────────────────────────────────────────────────
lang_choice = st.radio("", ["🇬🇧 English", "🇪🇬 العربية"], horizontal=True)
lang = "ar" if "العربية" in lang_choice else "en"
L = LANG[lang]
rtl_class = "rtl" if lang == "ar" else ""

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero {rtl_class}">
    <span class="hero-icon">💻</span>
    <h1>{L['title']}</h1>
    <p>{L['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ── Currency Selector ──────────────────────────────────────────────────────────
st.markdown(f'<div class="card {rtl_class}"><div class="card-title">{L["currency_label"]}</div>', unsafe_allow_html=True)
currency = st.selectbox("", list(CURRENCIES.keys()), label_visibility="collapsed")
rate = CURRENCIES[currency]
currency_code = currency.split(" ")[1]
st.markdown('</div>', unsafe_allow_html=True)

# ── Form Card ──────────────────────────────────────────────────────────────────
st.markdown(f'<div class="card {rtl_class}"><div class="card-title">{L["section"]}</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    brand   = st.selectbox(L['brand'],   le_brand.classes_)
    storage = st.selectbox(L['storage'], [128, 256, 512, 1024])
    cpu     = st.selectbox(L['cpu'],     [4,5,6,7,8,9,10,11])
with col2:
    condition = st.selectbox(L['condition'], le_condition.classes_)
    ram       = st.selectbox(L['ram'],       [4, 8, 16, 32])
    year      = st.selectbox(L['year'],      list(range(2018, 2027)))

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button(L['button']):
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

    mid_price = model.predict(new_data)[0]
    low_price  = mid_price * 0.88
    high_price = mid_price * 1.12

    # Convert
    low_c  = low_price  * rate
    mid_c  = mid_price  * rate
    high_c = high_price * rate

    fmt = lambda v: f"{v:,.0f}"

    st.markdown(f'<div class="result-title">{L["est_label"]}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="range-grid">
        <div class="range-box low">
            <div class="range-label">{L['low']}</div>
            <div class="range-price">{fmt(low_c)}</div>
            <div class="range-currency">{currency_code}</div>
        </div>
        <div class="range-box mid">
            <div class="range-label">{L['mid']}</div>
            <div class="range-price">{fmt(mid_c)}</div>
            <div class="range-currency">{currency_code}</div>
        </div>
        <div class="range-box high">
            <div class="range-label">{L['high']}</div>
            <div class="range-price">{fmt(high_c)}</div>
            <div class="range-currency">{currency_code}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="footer {rtl_class}">{L["footer"]}</div>', unsafe_allow_html=True)
