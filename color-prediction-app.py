import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from io import BytesIO

try:
    from sklearn.naive_bayes import MultinomialNB
except ModuleNotFoundError:
    st.error("❌ 'scikit-learn' not found. Please install it in requirements.txt.")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="🐉 Dragon Tiger Predictor AI", layout="centered")
st.title("🐉 Dragon vs ⚖️ Tiger Predictor (AI-Powered)")

st.markdown("""
    <style>
        body { background-color: #0f1117; color: white; }
        .stButton>button {
            background-color: #9c27b0;
            color: white;
            font-weight: bold;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Init ---
for key, default in {
    "authenticated": False,
    "username": "",
    "inputs": [],
    "X_train": [],
    "y_train": [],
    "log": [],
    "loss_streak": 0,
    "markov": defaultdict(lambda: defaultdict(int))
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Login Logic ---
def login(u, p): return p == "1234"

if not st.session_state.authenticated:
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(u, p):
            st.session_state.authenticated = True
            st.session_state.username = u
            st.success("✅ Logged in")
        else:
            st.error("❌ Invalid credentials")
    st.stop()

if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()

# --- Encoding ---
def encode(seq):
    m = {'D': 0, 'T': 1, 'TIE': 2}
    return [m[s] for s in seq if s in m]

def decode(val):
    m = {0: 'D', 1: 'T', 2: 'TIE'}
    return m.get(val, "")

# --- Prediction Logic ---
def predict(seq):
    if len(seq) < 10:
        return adaptive_fallback(seq)

    if len(st.session_state.X_train) < 20:
        return adaptive_fallback(seq)

    encoded = encode(seq[-10:])
    clf = MultinomialNB()
    weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
    clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
    pred = clf.predict([encoded])[0]
    conf = max(clf.predict_proba([encoded])[0]) * 100
    return decode(pred), round(conf)

def adaptive_fallback(seq):
    counts = {x: seq.count(x) for x in ['D', 'T', 'TIE']}
    best = max(counts, key=counts.get)
    return best, 60

# --- Learning Logic ---
def learn(seq, actual):
    if len(seq) >= 10:
        encoded_seq = encode(seq[-10:])
        st.session_state.X_train.append(encoded_seq)
        st.session_state.y_train.append(encode([actual])[0])
    for l in range(10, 4, -1):
        if len(seq) >= l:
            key = tuple(seq[-l:])
            st.session_state.markov[key][actual] += 1

# --- Input UI ---
st.subheader("🎮 Add Result (D / T / TIE)")
choice = st.selectbox("Latest Result", ["D", "T", "TIE"])
if st.button("➕ Add Result"):
    st.session_state.inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Continuous Learning from History ---
if len(st.session_state.inputs) > 10:
    for i in range(10, len(st.session_state.inputs)):
        hist = st.session_state.inputs[i-10:i]
        label = st.session_state.inputs[i]
        encoded = encode(hist)
        if len(encoded) == 10:
            st.session_state.X_train.append(encoded)
            st.session_state.y_train.append(encode([label])[0])

# --- Prediction Section ---
if len(st.session_state.inputs) >= 10:
    pred, conf = predict(st.session_state.inputs)

    st.subheader("📈 AI Prediction")
    if st.session_state.loss_streak >= 3:
        st.warning("⚠️ 3+ incorrect predictions in a row. Be cautious!")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

    st.success(f"Predicted: **{pred}** | Confidence: `{conf}%`")
    if conf >= 85:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
    else:
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)

    actual = st.selectbox("Enter Actual Result", ["D", "T", "TIE"], key="actual_feedback")
    if st.button("✅ Confirm & Learn"):
        correct = actual == pred
        st.session_state.log.append({
            "Prediction": pred,
            "Confidence": conf,
            "Actual": actual,
            "Correct": "✅" if correct else "❌"
        })
        learn(st.session_state.inputs, actual)
        st.session_state.inputs.append(actual)
        st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak + 1
        st.success("Learned and model updated.")
        st.rerun()
else:
    st.info(f"Enter {10 - len(st.session_state.inputs)} more rounds to enable prediction.")

# --- Training Info ---
if st.session_state.y_train:
    label_counts = pd.Series([decode(y) for y in st.session_state.y_train]).value_counts().to_dict()
    st.caption(f"📊 Training Summary: {label_counts}")

# --- History + Export ---
if st.session_state.log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df, use_container_width=True)

    if st.button("📥 Export to Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", data=buf.getvalue(),
                           file_name=f"{st.session_state.username}_dragon_tiger_history.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("🔍 Powered by Naive Bayes, Markov Patterns, and Time-Weighted Learning | Built with ❤️ by Vendra")
