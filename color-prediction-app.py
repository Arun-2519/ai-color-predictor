
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from io import BytesIO
import os
import pickle
from sklearn.naive_bayes import MultinomialNB

# --- UI Setup ---
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

# --- Session Setup ---
def init_session():
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
init_session()

# --- Data Directory ---
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
                    saved = pickle.load(f)
                    st.session_state.X_train = saved.get("X_train", [])
                    st.session_state.y_train = saved.get("y_train", [])
            LOG_PATH = os.path.join(DATA_DIR, f"{u}_log.csv")
            if os.path.exists(LOG_PATH):
                df = pd.read_csv(LOG_PATH)
                st.session_state.prediction_log = df.to_dict(orient="records")
        else:
            st.error("❌ Invalid login")
    st.stop()

if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.rerun()

# --- Encoding ---
def encode(seq):
    m = {"Red": 0, "Black": 1, "Joker": 2}
    return [m.get(s, 0) for s in seq]

def decode(val):
    m = {0: "Red", 1: "Black", 2: "Joker"}
    return m.get(val, "")

# --- Input Section ---
st.subheader("🎮 Add Game Result")
choice = st.selectbox("Latest Result", ["Red", "Black", "Joker"])
if st.button("➕ Add Result"):
    st.session_state.user_inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Learning From Input History ---
if len(st.session_state.user_inputs) > 8:
    for i in range(8, len(st.session_state.user_inputs)):
        past = st.session_state.user_inputs[i - 8:i]
        next_val = st.session_state.user_inputs[i]
        enc = encode(past)
        if len(enc) == 8:
            st.session_state.X_train.append(enc)
            st.session_state.y_train.append(encode([next_val])[0])
            if len(st.session_state.X_train) > 3000:
                st.session_state.X_train = st.session_state.X_train[-3000:]
                st.session_state.y_train = st.session_state.y_train[-3000:]

# --- Learning Function ---
def learn(seq, actual):
    if len(seq) >= 8:
        encoded = encode(seq[-8:])
        label = encode([actual])[0]
        st.session_state.X_train.append(encoded)
        st.session_state.y_train.append(label)
    for l in range(8, 3, -1):
        key = tuple(seq[-l:])
        st.session_state.transition_model[key][actual] += 1

# --- Pattern Matching ---
def match_partial_pattern(seq):
    for l in range(8, 3, -1):
        current = tuple(seq[-l:])
        matches = {k: v for k, v in st.session_state.transition_model.items() if k[-l:] == current}
        if matches:
            counter = Counter()
            for v in matches.values():
                counter.update(v)
            total = sum(counter.values())
            if total > 0:
                return {k: round((v / total) * 100, 1) for k, v in counter.items()}
    return {}

# --- Fallback: Trend of Last 10 ---
def recent_trend_prediction(seq):
    reds = seq[-10:].count("Red")
    blacks = seq[-10:].count("Black")
    jokers = seq[-10:].count("Joker")
    counts = {"Red": reds, "Black": blacks, "Joker": jokers}
    top = max(counts, key=counts.get)
    return top, 55

# --- Prediction Core ---
def predict_color(seq):
    if len(seq) < 10:
        return "Learning", 0

    pattern_probs = match_partial_pattern(seq)
    if pattern_probs:
        top = max(pattern_probs, key=pattern_probs.get)
        return top, pattern_probs[top]

    top10, conf10 = recent_trend_prediction(seq)
    if conf10 > 55:
        return top10, conf10

    if len(st.session_state.X_train) >= 20:
        model = MultinomialNB()
        encoded = encode(seq[-8:])
        if len(encoded) < 8:
            encoded = [0] * (8 - len(encoded)) + encoded
        weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
        model.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
        pred = model.predict([encoded])[0]
        conf = max(model.predict_proba([encoded])[0]) * 100
        return decode(pred), round(conf)

    return "Learning", 0

# --- Prediction & Feedback UI ---
if len(st.session_state.user_inputs) >= 10:
    pred, conf = predict_color(st.session_state.user_inputs)
    st.subheader("📈 AI Prediction")

    if st.session_state.loss_streak >= 3:
        st.warning("⚠️ 3+ wrong predictions in a row.")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

    if conf < 60:
        st.warning(f"⛔ Low confidence: **{pred}** ({conf}%)")
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)
    else:
        st.success(f"✅ Predicted: **{pred}** | Confidence: `{conf}%`")
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
    st.info(f"🔁 Enter {10 - len(st.session_state.user_inputs)} more results to begin prediction.")

if st.session_state.y_train:
    label_counts = pd.Series([decode(y) for y in st.session_state.y_train]).value_counts().to_dict()
    st.caption(f"📊 Training Summary: {label_counts}")

if st.session_state.prediction_log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(df, use_container_width=True)
    if st.button("📥 Export Excel"):
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button("⬇️ Download Excel", data=buffer.getvalue(),
            file_name=f"{st.session_state.username}_prediction_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("Built with ❤️ | Adaptive + Pattern Learning | Sound Alerts")
