import streamlit as st
import pandas as pd
from collections import defaultdict
from io import BytesIO
import random
import numpy as np
from sklearn.naive_bayes import MultinomialNB

# --- Streamlit UI config ---
st.set_page_config(page_title="🧠 AI Color Predictor", layout="centered")
st.markdown("""
    <style>
        body { background-color: #0f1117; color: #ffffff; }
        .stButton>button {
            background-color: #6a1b9a;
            color: white;
            border-radius: 6px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Color Predictor with Bayesian & Pattern Learning")

# --- Session Setup ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []
if "transition_model" not in st.session_state:
    st.session_state.transition_model = defaultdict(lambda: defaultdict(int))
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0
if "X_train" not in st.session_state:
    st.session_state.X_train = []
if "y_train" not in st.session_state:
    st.session_state.y_train = []

# --- Login System ---
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

# --- Add Input Interface ---
st.subheader("🎮 Add Round Result (Red / Black / Joker)")
choice = st.selectbox("Latest Result", ["Red", "Black", "Joker"])
if st.button("➕ Add Result"):
    st.session_state.user_inputs.append(choice)
    st.success(f"Added: {choice}")

# --- Encoding ---
def encode_sequence(seq):
    mapping = {"Red": 0, "Black": 1, "Joker": 2}
    return [mapping[s] for s in seq if s in mapping]

def decode_label(val):
    rev_map = {0: "Red", 1: "Black", 2: "Joker"}
    return rev_map.get(val, "")

# --- Prediction Logic ---
def predict_color(sequence):
    if len(sequence) < 8:
        return adaptive_fallback(sequence)

    seq = encode_sequence(sequence[-8:])

    if len(st.session_state.X_train) >= 20:
        clf = MultinomialNB()
        weights = np.exp(np.linspace(0, 1, len(st.session_state.X_train)))
        clf.fit(st.session_state.X_train, st.session_state.y_train, sample_weight=weights)
        pred = clf.predict([seq])[0]
        conf = max(clf.predict_proba([seq])[0]) * 100
        return decode_label(pred), round(conf)

    return adaptive_fallback(sequence)

def adaptive_fallback(sequence):
    reds = sequence[-10:].count("Red")
    blacks = sequence[-10:].count("Black")
    if reds > blacks:
        return "Black", 70
    elif blacks > reds:
        return "Red", 70
    else:
        return random.choice(["Red", "Black"]), 60

# --- Model Learning ---
def update_learning(sequence, next_color):
    if len(sequence) >= 8:
        encoded = encode_sequence(sequence[-8:])
        label = encode_sequence([next_color])[0]
        st.session_state.X_train.append(encoded)
        st.session_state.y_train.append(label)
    for length in range(8, 4, -1):
        if len(sequence) >= length:
            key = tuple(sequence[-length:])
            st.session_state.transition_model[key][next_color] += 1

# --- Prediction Section ---
if len(st.session_state.user_inputs) >= 10:
    prediction, conf = predict_color(st.session_state.user_inputs)

    st.subheader("📈 AI Prediction")

    if st.session_state.loss_streak >= 3:
        st.warning("⚠️ More than 3 incorrect predictions in a row. Please proceed with caution.")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)

    st.success(f"Predicted: **{prediction}** | Confidence: `{conf}%`")
    if conf >= 85:
        st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", autoplay=True)
    else:
        st.audio("https://actions.google.com/sounds/v1/alarms/warning.ogg", autoplay=True)

    actual = st.selectbox("Enter actual result:", ["Red", "Black", "Joker"], key="actual_feedback")
    if st.button("✅ Confirm & Learn"):
        correct = (actual == prediction)
        st.session_state.prediction_log.append({
            "Prediction": prediction,
            "Confidence": conf,
            "Actual": actual,
            "Correct": "✅" if correct else "❌"
        })
        update_learning(st.session_state.user_inputs, actual)
        st.session_state.user_inputs.append(actual)

        if correct:
            st.session_state.loss_streak = 0
        else:
            st.session_state.loss_streak += 1

        st.success("Learned and updated model.")
        st.rerun()
else:
    st.warning(f"Need {10 - len(st.session_state.user_inputs)} more entries to start prediction.")

# --- History & Export Section ---
if st.session_state.prediction_log:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.prediction_log)
    st.dataframe(df, use_container_width=True)

    if st.button("📥 Generate Excel"):
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button("⬇️ Download Excel", buffer, file_name=f"{st.session_state.username}_AI_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
st.caption("Built with ❤️ | Naive Bayes + Markov Pattern + Time Weighted | TensorFlow-Free Edition")
