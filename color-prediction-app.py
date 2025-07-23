import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from io import BytesIO
from sklearn.naive_bayes import MultinomialNB

# --- UI Setup ---
st.set_page_config(page_title="🧠 AI Color Predictor", layout="centered")
st.title("🧠 Color Predictor (Red / Black / Joker) | AI Powered")

st.markdown("""
    <style>
        body { background-color: #0f1117; color: white; }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
            border-radius: 6px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
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

# --- Encoding Functions ---
def encode(seq):
    m = {"Red": 0, "Black": 1, "Joker": 2}
    return [m[s] for s in seq if s in m]

def decode(val):
    m = {0: "Red", 1: "Black", 2: "Joker"}
    return m.get(val, "")

# --- Input UI ---
st.subheader("🎮 Add Round Result (Red / Black / Joker)")
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

# --- Fallback ---
def adaptive_fallback(seq):
    reds = seq[-10:].count("Red")
    blacks = seq[-10:].count("Black")
    jokers = seq[-10:].count("Joker")
    counts = {"Red": reds, "Black": blacks, "Joker": jokers}
    best = max(counts, key=counts.get)
    return best, 60

# --- Learn Function ---
def learn(sequence, actual):
    if len(sequence) >= 8:
        enc = encode(sequence[-8:])
        label = encode([actual])[0]
        st.session_state.X_train.append(enc)
        st.session_state.y_train.append(label)
    for l in range(8, 4, -1):
        if len(sequence) >= l:
            key = tuple(sequence[-l:])
            st.session_state.transition_model[key][actual] += 1

# --- Prediction UI ---
if len(st.session_state.user_inputs) >= 10:
    prediction, conf = predict_color(st.session_state.user_inputs)

    st.subheader("📈 AI Prediction")
    if st.session_state.loss_streak >= 3:
        st.warning("⚠️ More than 3 wrong predictions in a row. Be cautious!")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

    st.success(f"Predicted: **{prediction}** | Confidence: `{conf}%`")
    if conf >= 85:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
    else:
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)

    actual = st.selectbox("Enter Actual Result:", ["Red", "Black", "Joker"], key="actual_feedback")
    if st.button("✅ Confirm & Learn"):
        correct = (actual == prediction)
        st.session_state.prediction_log.append({
            "Prediction": prediction,
            "Confidence": conf,
            "Actual": actual,
            "Correct": "✅" if correct else "❌"
        })
        learn(st.session_state.user_inputs, actual)
        st.session_state.user_inputs.append(actual)
        st.session_state.loss_streak = 0 if correct else st.session_state.loss_streak + 1
        st.success("Learned and model updated.")
        st.rerun()
else:
    st.info(f"Need {10 - len(st.session_state.user_inputs)} more rounds to predict.")

# --- Training Summary ---
if st.session_state.y_train:
    label_counts = pd.Series([decode(y) for y in st.session_state.y_train]).value_counts().to_dict()
    st.caption(f"📊 Training Samples: {label_counts}")

# --- History & Export ---
if st.session_state.prediction_log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(df, use_container_width=True)

    if st.button("📥 Export to Excel"):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        st.download_button("⬇️ Download Excel", data=buf.getvalue(),
                           file_name=f"{st.session_state.username}_color_history.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("🔮 Powered by Naive Bayes, Pattern Memory & Adaptive Fallback | Built with ❤️ by Vendra")
