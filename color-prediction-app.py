# --- imports ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict, Counter
from io import BytesIO
from sklearn.naive_bayes import MultinomialNB

# --- UI Config ---
st.set_page_config(page_title="🎨 AI Color Predictor", layout="centered")
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
def init_session():
    for key, val in {
        "authenticated": False, "username": "", "user_inputs": [],
        "X_train": [], "y_train": [], "prediction_log": [],
        "loss_streak": 0, "transition_model": defaultdict(lambda: defaultdict(int))
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val
init_session()

# --- File Paths ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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

            TRAIN_PATH = os.path.join(DATA_DIR, f"{u}_training.pkl")
            if os.path.exists(TRAIN_PATH):
                with open(TRAIN_PATH, "rb") as f:
                    d = pickle.load(f)
                    st.session_state.X_train = d.get("X_train", [])
                    st.session_state.y_train = d.get("y_train", [])
            LOG_PATH = os.path.join(DATA_DIR, f"{u}_log.csv")
            if os.path.exists(LOG_PATH):
                df = pd.read_csv(LOG_PATH)
                st.session_state.prediction_log = df.to_dict(orient="records")
        else:
            st.error("❌ Invalid login")
    st.stop()

if st.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- Encoders ---
def encode(seq): return [ {"Red": 0, "Black": 1, "Joker": 2}[s] for s in seq if s in {"Red", "Black", "Joker"} ]
def decode(val): return {0: "Red", 1: "Black", 2: "Joker"}.get(val, "")

# --- Input ---
st.subheader("🎮 Add Game Result (Red / Black / Joker)")
choice = st.selectbox("Latest Result", ["Red", "Black", "Joker"])
if st.button("➕ Add Result"):
    st.session_state.user_inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Learn from User Inputs ---
if len(st.session_state.user_inputs) > 8:
    for i in range(8, len(st.session_state.user_inputs)):
        past = st.session_state.user_inputs[i - 8:i]
        next_val = st.session_state.user_inputs[i]
        enc = encode(past)
        if len(enc) == 8:
            st.session_state.X_train.append(enc)
            st.session_state.y_train.append(encode([next_val])[0])
            # Trim model size
            if len(st.session_state.X_train) > 3000:
                st.session_state.X_train = st.session_state.X_train[-3000:]
                st.session_state.y_train = st.session_state.y_train[-3000:]

# --- Pattern Matching ---
def match_partial_pattern(seq):
    for l in range(8, 3, -1):
        current = tuple(seq[-l:])
        matches = {k: v for k, v in st.session_state.transition_model.items() if k[-l:] == current}
        if matches:
            counter = Counter()
            for v in matches.values(): counter.update(v)
            total = sum(counter.values())
            if total > 0:
                return {k: round((v / total) * 100, 1) for k, v in counter.items()}
    return {}

# --- Trend fallback (recent 10) ---
def recent_trend_prediction(seq):
    reds = seq[-10:].count("Red")
    blacks = seq[-10:].count("Black")
    jokers = seq[-10:].count("Joker")
    counts = {"Red": reds, "Black": blacks, "Joker": jokers}
    top = max(counts, key=counts.get)
    return top, 55

# --- Learn Function ---
def learn(seq, actual):
    if len(seq) >= 8:
        st.session_state.X_train.append(encode(seq[-8:]))
        st.session_state.y_train.append(encode([actual])[0])
    for l in range(8, 3, -1):
        key = tuple(seq[-l:])
        st.session_state.transition_model[key][actual] += 1

# --- Prediction ---
def predict_color(seq):
    if len(seq) < 4:
        return "Waiting", 0

    # 1. Pattern Match
    pattern_probs = match_partial_pattern(seq)
    if pattern_probs:
        top = max(pattern_probs, key=pattern_probs.get)
        return top, pattern_probs[top]

    # 2. Trend
    if len(seq) >= 10:
        top10, conf10 = recent_trend_prediction(seq)
        return top10, conf10

    # 3. Naive Bayes
    if len(st.session_state.X_train) >= 20:
        model = MultinomialNB()
        encoded = encode(seq[-8:])
        weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
        model.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
        pred = model.predict([encoded])[0]
        conf = max(model.predict_proba([encoded])[0]) * 100
        return decode(pred), round(conf)

    return "Learning", 0

# --- Prediction UI ---
if len(st.session_state.user_inputs) >= 4:
    pred, conf = predict_color(st.session_state.user_inputs)
    st.subheader("📈 AI Prediction")

    if st.session_state.loss_streak >= 3:
        st.warning("🚨 3+ Wrong predictions. Take caution.")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

    if conf == 0:
        st.info("🌀 Still learning... Enter more inputs.")
    elif conf < 60:
        st.warning(f"⛔ Low confidence: **{pred}** ({conf}%)")
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
    else:
        st.success(f"🎯 Predicted: **{pred}** | Confidence: `{conf}%`")
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)

        actual = st.selectbox("Enter Actual Result:", ["Red", "Black", "Joker"], key="actual_feedback")
        if st.button("✅ Confirm & Learn"):
            correct = (actual == pred)
            st.session_state.prediction_log.append({
                "Prediction": pred, "Confidence": conf, "Actual": actual,
                "Correct": "✅" if correct else "❌"
            })
            learn(st.session_state.user_inputs, actual)
            st.session_state.user_inputs.append(actual)
            st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak + 1

            # Save
            pickle.dump(
                {"X_train": st.session_state.X_train, "y_train": st.session_state.y_train},
                open(os.path.join(DATA_DIR, f"{st.session_state.username}_training.pkl"), "wb")
            )
            pd.DataFrame(st.session_state.prediction_log).to_csv(
                os.path.join(DATA_DIR, f"{st.session_state.username}_log.csv"), index=False
            )
            st.success("✅ Learned and updated.")
            st.rerun()
else:
    st.info("Enter at least 4 rounds to begin prediction.")

# --- Stats and Download ---
if st.session_state.y_train:
    counts = pd.Series([decode(y) for y in st.session_state.y_train]).value_counts().to_dict()
    st.caption(f"📊 Training Data Breakdown: {counts}")
if st.session_state.prediction_log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(df, use_container_width=True)

    if st.button("📥 Export Excel"):
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button("⬇️ Download", data=buffer.getvalue(),
            file_name=f"{st.session_state.username}_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("🔁 Real-time Pattern & Streak Learning | 🔊 Sound Enabled")
