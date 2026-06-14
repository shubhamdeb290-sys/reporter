import streamlit as st
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

# Sample dataset (no CSV needed)
emails = [
    "Win a free iPhone now!!!",
    "Meeting scheduled at 3 PM tomorrow",
    "Congratulations! You won a lottery, claim now",
    "Project report submission deadline is Monday",
    "Get cheap loans instantly, apply today!",
    "Lunch with team at 1 PM",
    "Exclusive offer just for you, click here!"
]
labels = [1, 0, 1, 0, 1, 0, 1]  # 1 = spam, 0 = ham

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(emails)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
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
