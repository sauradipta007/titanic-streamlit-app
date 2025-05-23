import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Upload Titanic data and predict passenger survival!")

# Load and preprocess training data
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    train['Age'].fillna(train['Age'].median(), inplace=True)
    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
    train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return train

train = load_data()

# Train model
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
acc = accuracy_score(y_val, model.predict(X_val))

st.subheader("Model Accuracy")
st.success(f"✅ Accuracy on validation set: {acc:.2f}")

# Visualize
st.subheader("Survival by Sex")
fig, ax = plt.subplots()
sns.countplot(x='Survived', hue='Sex', data=train, ax=ax)
st.pyplot(fig)

# Upload and predict
st.subheader("📁 Upload Test Data to Predict Survival")
uploaded_file = st.file_uploader("Upload a test CSV file", type="csv")

if uploaded_file:
    test = pd.read_csv(uploaded_file)
    test['Age'].fillna(train['Age'].median(), inplace=True)
    test['Fare'].fillna(train['Fare'].median(), inplace=True)
    test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
    test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    test.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

    predictions = model.predict(test.drop('PassengerId', axis=1))
    test['Survived'] = predictions
    st.write("✅ Predictions:")
    st.dataframe(test[['PassengerId', 'Survived']])

    csv = test[['PassengerId', 'Survived']].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction CSV", data=csv, file_name="submission.csv", mime='text/csv')
