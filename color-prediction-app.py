import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from io import BytesIO
from sklearn.naive_bayes import MultinomialNB

# --- UI Setup ---
st.set_page_config(page_title="🎨 Color Predictor AI", layout="centered")
st.title("🧠 AI Color Predictor (Red / Black / Joker)")

st.markdown("""
    <style>
        body { background-color: #0f1117; color: white; }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
            font-weight: bold;
            border-radius: 6px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session Init ---
for key, default in {
    "authenticated": False,
    "username": "",
    "user_inputs": [],
    "X_train": [],
    "y_train": [],
    "prediction_log": [],
    "loss_streak": 0,
    "transition_model": defaultdict(lambda: defaultdict(int))
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Login ---
def login(user, pwd): return pwd == "1234"
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
            st.error("❌ Invalid login")
    st.stop()
if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()

# --- Encoders ---
def encode(seq):
    m = {"Red": 0, "Black": 1, "Joker": 2}
    return [m[s] for s in seq if s in m]

def decode(val):
    m = {0: "Red", 1: "Black", 2: "Joker"}
    return m.get(val, "")

# --- Input Interface ---
st.subheader("🎮 Add Game Result (Red / Black / Joker)")
choice = st.selectbox("Latest Result", ["Red", "Black", "Joker"])
if st.button("➕ Add Result"):
    st.session_state.user_inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Continuous Learning ---
if len(st.session_state.user_inputs) > 8:
    for i in range(8, len(st.session_state.user_inputs)):
        past = st.session_state.user_inputs[i-8:i]
        next_val = st.session_state.user_inputs[i]
        enc = encode(past)
        if len(enc) == 8:
            st.session_state.X_train.append(enc)
            st.session_state.y_train.append(encode([next_val])[0])

# --- Prediction Logic ---
def predict_color(seq):
    if len(seq) < 8:
        return adaptive_fallback(seq)

    encoded = encode(seq[-8:])
    if len(st.session_state.X_train) < 20:
        return adaptive_fallback(seq)

    clf = MultinomialNB()
    weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
    clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
    pred = clf.predict([encoded])[0]
    conf = max(clf.predict_proba([encoded])[0]) * 100
    return decode(pred), round(conf)

def adaptive_fallback(seq):
    reds = seq[-10:].count("Red")
    blacks = seq[-10:].count("Black")
    jokers = seq[-10:].count("Joker")
    counts = {"Red": reds, "Black": blacks, "Joker": jokers}
    best = max(counts, key=counts.get)
    return best, 55

# --- Learning Logic ---
def learn(seq, actual):
    if len(seq) >= 8:
        encoded = encode(seq[-8:])
        label = encode([actual])[0]
        st.session_state.X_train.append(encoded)
        st.session_state.y_train.append(label)
    for l in range(8, 4, -1):
        if len(seq) >= l:
            key = tuple(seq[-l:])
            st.session_state.transition_model[key][actual] += 1

# --- Prediction Section ---
if len(st.session_state.user_inputs) >= 10:
    pred, conf = predict_color(st.session_state.user_inputs)

    st.subheader("📈 AI Prediction")

    if st.session_state.loss_streak >= 3:
        st.warning("⚠️ 3+ wrong predictions in a row. Wait for better pattern!")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

    if conf < 65:
        st.warning("⛔ Low confidence or limited data. Wait...")
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
    else:
        st.success(f"Predicted: **{pred}** | Confidence: `{conf}%`")
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)

        actual = st.selectbox("Enter Actual Result:", ["Red", "Black", "Joker"], key="actual_feedback")
        if st.button("✅ Confirm & Learn"):
            correct = (actual == pred)
            st.session_state.prediction_log.append({
                "Prediction": pred,
                "Confidence": conf,
                "Actual": actual,
                "Correct": "✅" if correct else "❌"
            })
            learn(st.session_state.user_inputs, actual)
            st.session_state.user_inputs.append(actual)
            st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak + 1
            st.success("✅ Learned and model updated.")
            st.rerun()
else:
    st.info(f"Enter {10 - len(st.session_state.user_inputs)} more results to enable prediction.")

# --- Training Stats ---
if st.session_state.y_train:
    label_counts = pd.Series([decode(y) for y in st.session_state.y_train]).value_counts().to_dict()
    st.caption(f"📊 Training Distribution: {label_counts}")

# --- History + Export ---
if st.session_state.prediction_log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(df, use_container_width=True)

    if st.button("📥 Generate Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", data=buf.getvalue(),
                           file_name=f"{st.session_state.username}_color_prediction_history.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("🔮 AI-Powered Color Prediction App | Naive Bayes + Confidence Alerts | Built with ❤️ by Vendra")
