import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('emails.csv')
    return data

# Train the model
@st.cache_resource
def train_model(data):
    X = data['Message']
    y = data['Category']

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return clf, vectorizer, accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

# Streamlit App UI
st.title("Email Spam Classifier")

# Load dataset and train the model
st.sidebar.subheader("Model Training Info")
data = load_data()
clf, vectorizer, accuracy, report, confusion = train_model(data)

# Display dataset
if st.sidebar.checkbox("Show Dataset"):
    st.subheader("Dataset Preview")
    st.write(data.head())

# Display model performance
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")
if st.sidebar.checkbox("Show Classification Report"):
    st.subheader("Classification Report")
    st.text(report)

if st.sidebar.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix")
    st.write(confusion)

# Prediction Section
st.subheader("Email Spam Prediction")
user_input = st.text_area("Enter the email content:")
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter an email content!")
    else:
        # Vectorize and predict
        user_email_vec = vectorizer.transform([user_input])
        prediction = clf.predict(user_email_vec)
        result = "Spam" if prediction[0] == "spam" else "Not Spam"
        st.success(f"The email is classified as: {result}")
