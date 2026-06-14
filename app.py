import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

st.title("📧 Email Spam Detection App")

# Expanded sample dataset
emails = [
    "Win a free iPhone now!!!",
    "Meeting scheduled at 3 PM tomorrow",
    "Congratulations! You won a lottery, claim now",
    "Project report submission deadline is Monday",
    "Get cheap loans instantly, apply today!",
    "Lunch with team at 1 PM",
    "Exclusive offer just for you, click here!",
    "Your order has been shipped",
    "Can we reschedule our call?",
    "Happy birthday! Have a great day"
]
labels = [1,0,1,0,1,0,1,0,0,0]  # 1 = spam, 0 = ham

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

# User input for multiple messages
st.subheader("Test multiple messages:")
multi_input = st.text_area("Enter messages separated by new lines:")

if st.button("Check Messages"):
    messages = [m.strip() for m in multi_input.split("\n") if m.strip()]
    if messages:
        for msg in messages:
            pred = model.predict(vectorizer.transform([msg]))[0]
            if pred == 1:
                st.error(f"🚨 SPAM → {msg}")
            else:
                st.success(f"📩 HAM → {msg}")
    else:
        st.warning("Please enter at least one message.")
