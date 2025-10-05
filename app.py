import streamlit as st
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="AI-Powered Mental Health Journal Analyzer 💬🧠", page_icon="💖", layout="wide")

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    dataset = load_dataset("dair-ai/emotion", split="train")
    df = pd.DataFrame(dataset)
    return df["text"], df["label"]

texts, labels = load_data()

# --- TRAIN MODEL ---
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)
    return model, vectorizer

model, vectorizer = train_model()

# --- APP TITLE ---
st.title("💬 AI-Powered Mental Health Journal Analyzer")
st.caption("Created by **Amulya Adepu (236Y1A6604)** 🌸")

st.write("This intelligent journaling app analyzes your daily thoughts and emotions, providing insights and supportive suggestions to boost your emotional well-being.")

st.divider()

# --- USER INPUT ---
user_input = st.text_area("📝 Write your journal entry below:", placeholder="Type your thoughts here...", height=150)

# --- SESSION STATE FOR PERSONAL HISTORY ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- EMOTION LABELS & SUGGESTIONS ---
emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
suggestions = {
    "sadness": [
        "Try writing three things you’re grateful for today 🌸",
        "Take a warm shower, listen to calming music, and let your emotions flow 💧",
        "Reach out to someone who makes you feel seen — connection heals 💕"
    ],
    "joy": [
        "Share your happiness with someone close! 😊",
        "Capture this moment — write down why you feel so good 🌼",
        "Celebrate small wins — they matter more than you think 🎉"
    ],
    "love": [
        "Write a heartfelt message to someone you appreciate 💌",
        "Spend time doing something kind for yourself or others 💖",
        "Express gratitude — love grows when shared 🌷"
    ],
    "anger": [
        "Pause and take slow breaths — count to ten before reacting 😌",
        "Go for a walk or stretch it out — movement helps release tension 🧘",
        "Write down what triggered you and reflect on what you can control 🌿"
    ],
    "fear": [
        "Focus on your breathing — you’re safe in this moment 🌱",
        "Write about what’s worrying you, then list things you *can* control 🌤️",
        "Try grounding: name 5 things you see, 4 touch, 3 hear, 2 smell, 1 taste 🌍"
    ],
    "surprise": [
        "Reflect — was it a good or bad surprise? How did it make you feel? 🤔",
        "Sometimes unexpected things bring new perspectives — stay open 🌈",
        "Write what you learned from today’s surprise moment 🌟"
    ]
}

# --- ANALYSIS BUTTON ---
if st.button("🔍 Analyze My Emotions"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        predicted_emotion = emotions[prediction]
        chosen_suggestion = random.choice(suggestions[predicted_emotion])

        # Add to personal history
        st.session_state["history"].append({
            "text": user_input,
            "emotion": predicted_emotion,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # Display result
        st.success(f"💭 **Predicted Emotion:** {predicted_emotion.upper()}")
        st.info(chosen_suggestion)
    else:
        st.warning("Please write something first 💬")

# --- USER HISTORY ---
if st.session_state["history"]:
    st.subheader("🕓 Your Personal Emotion Journal")
    df_hist = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_hist[["time", "text", "emotion"]].sort_values(by="time", ascending=False), use_container_width=True)

    # --- GRAPHS ---
    st.subheader("📊 Emotion Insights")
    col1, col2 = st.columns(2)

    # Emotion Distribution
    with col1:
        counts = df_hist["emotion"].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", color="pink", ax=ax)
        ax.set_title("Emotion Distribution 💖")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # Emotion Over Time
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(df_hist["time"], df_hist["emotion"], marker="o", color="deeppink")
        ax2.set_title("Emotion Over Time 🌸")
        ax2.set_xlabel("Entry Time")
        ax2.set_ylabel("Emotion")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
