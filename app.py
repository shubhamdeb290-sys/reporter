import streamlit as st
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources (fixes deployment errors)
nltk.download('stopwords')
nltk.download('punkt')

# Title
st.title("📧 Email Spam Detection App")

# Load dataset (replace with your own CSV if needed)
@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
    data.columns = ['label','message']
    data['label'] = data['label'].map({'ham':0, 'spam':1})
    return data

data = load_data()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
st.write(f"✅ Model trained with accuracy: {acc:.2f}")

# User input
st.subheader("Try it yourself:")
user_input = st.text_area("Enter an email message:")

if st.button("Check Spam"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.error("🚨 This looks like SPAM!")
        else:
            st.success("📩 This looks like HAM (not spam).")
    else:
        st.warning("Please enter a message first.")
