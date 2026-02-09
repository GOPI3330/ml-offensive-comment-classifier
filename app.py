import streamlit as st
import pandas as pd
import pickle
import re
import string
from langdetect import detect
from googletrans import Translator

# Loading model
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder_label.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Text Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# App Title
st.title("Multilingual Offensive Comment Classifier (ML-OCC)🌐🚫💬 ")
st.markdown("This project detects offensive comments in South Indian Languages 🌍")

# Sidebar Navigation
option = st.sidebar.selectbox("Choose Section", [
    "📊 Data Samples",
    "🧹 Preprocessing Demo",
    "📈 Model Performance",
    "📝 Try Prediction"
])

# Translator for (#4prediction)
translator = Translator()

# Mapping language codes to full names
lang_names = {
    'ta': 'Tamil',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'en': 'English',
    'unknown': 'Unknown'
}

# Emoji for labels
emoji_map = {
    'not_offensive': "🙂",
    'offensive': "⚠️",
    'mixed_feelings': "😐",
    'Unknown': "❓"
}

# 1. Data Samples
if option == "📊 Data Samples":
    st.subheader("Training Data Preview")
    train_df = pd.read_csv("train_data.csv")
    st.dataframe(train_df.sample(5))

    st.subheader("Test Data Preview")
    test_df = pd.read_csv("test_data.csv")
    st.dataframe(test_df.sample(5))

# 2. Preprocessing
elif option == "🧹 Preprocessing Demo":
    st.subheader("Text Cleaning Example")
    sample = st.text_area("Enter a raw comment:")
    if sample:
        cleaned = clean_text(sample)
        st.write("✅ Cleaned Text:")
        st.success(cleaned)

# 3. Model Performance
elif option == "📈 Model Performance":
    st.subheader("Model Evaluation on Test Set")

    test_df = pd.read_csv("test_data.csv")

    if 'clean_text' not in test_df.columns:
        st.error("The column 'clean_text' is missing in test_data.csv. Please ensure it's preprocessed.")
    else:
        X_test = vectorizer.transform(test_df['clean_text'].fillna(""))
        y_test = test_df['label_encoded']
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        acc = accuracy_score(y_test, y_pred)

        st.metric("Test Accuracy", f"{acc:.2%}")
        st.dataframe(pd.DataFrame(report).transpose())

# 4. Prediction
elif option == "📝 Try Prediction":
    st.subheader("📝 Classify a Comment (Multilingual Support)")
    user_input = st.text_input("Enter your comment (Tamil, Malayalam, Kannada, or English):")

    if user_input:
        # Detect language
        try:
            lang = detect(user_input)
        except:
            lang = 'Unknown'

        # Translate to English (if not already English)
        if lang != 'en' and lang != 'Unknown':
            try:
                translated = translator.translate(user_input, src=lang, dest='en').text
            except:
                translated = "(Translation failed)"
        else:
            translated = user_input

        # Clean and vectorize input
        cleaned_input = clean_text(translated)
        vectorized_input = vectorizer.transform([cleaned_input])

        # Predict label
        prediction = model.predict(vectorized_input)[0]
        try:
            label = label_encoder.inverse_transform([prediction])[0]
        except:
            label = prediction

        emoji = emoji_map.get(label, "❓")

        st.success(f"Predicted Label: **{label}** {emoji}")
        st.info(f"🌐 Detected Language: **{lang_names.get(lang, 'Unknown')}** (`{lang}`)")
        st.info(f"📘 Translation: _\"{translated}\"_")

