# 💬 AI-Powered Mental Health Journal Analyzer

### 🧠 Created by: Amulya Adepu  
**Hall Ticket No:** 236Y1A6604  

---

## 🌟 Overview
This project helps users reflect on their mental health by analyzing daily journal entries using **AI emotion detection**.  
It identifies emotions from text (like joy, sadness, anger, fear, love, surprise) and gives supportive, personalized suggestions.

---

## ⚙️ How It Works

### 1️⃣ Input
The user writes a daily journal entry in a text box.

### 2️⃣ Processing
- Uses the **“dair-ai/emotion”** dataset for emotion-labeled text.
- Applies **TF-IDF vectorization** and trains a **Logistic Regression model**.
- Predicts the emotion of the new journal entry.

### 3️⃣ Output
- Displays the **predicted emotion**.
- Provides a **personalized, casual suggestion**.
- Saves the user’s entries **privately** (only they can view their own history).
- Generates two emotion insights:
  - 📊 **Emotion Distribution Bar Graph**
  - 🌸 **Emotion Over Time Line Graph**

---

## 🧩 Technologies Used
- **Python**
- **Streamlit** – Frontend web app framework
- **scikit-learn** – ML model training
- **datasets (HuggingFace)** – for loading dair-ai/emotion dataset
- **matplotlib** – for visualizations
- **pyngrok** – for temporary public access in Colab

---

## 🚀 Deployment
You can deploy this app easily on **Streamlit Cloud**:
1. Upload `app.py`, `requirements.txt`, and `README.md` to a **GitHub repo**.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Deploy the app — your personal link will be live in a few minutes!

---

## ❤️ Features
✅ Emotion detection from free text  
✅ Dynamic, empathetic suggestions  
✅ Private user history  
✅ Emotion distribution and trend graphs  
✅ Clean and calming UI with pink theme  

---

## 📊 Example Output
| Journal Entry | Predicted Emotion | Suggestion |
|----------------|-------------------|-------------|
| “Work was stressful but I managed to finish everything.” | Anger | Take a break and stretch — you did well under pressure! |
| “I had a great day with my friends today.” | Joy | Capture this moment — write down what made you happy today. |

---

## 👩‍💻 Credits
Dataset: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)  
Developed by **Amulya Adepu (236Y1A6604)** 🌸  
