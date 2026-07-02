import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction")

# Load dataset
df = pd.read_csv("titanicsurvival.csv")

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Features and Target
X = df[["Pclass", "Gender", "Age", "Fare"]]
y = df["Survived"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

gender = st.selectbox("Gender", ["male", "female"])
gender = le.transform([gender])[0]

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)

fare = st.number_input("Fare", min_value=0.0, value=30.0)

if st.button("Predict"):

    data = pd.DataFrame({
        "Pclass": [pclass],
        "Gender": [gender],
        "Age": [age],
        "Fare": [fare]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    if prediction == 1:
        st.success("✅ Passenger is likely to Survive")
    else:
        st.error("❌ Passenger is not likely to Survive")

    st.write(f"Survival Probability: **{probability[1]*100:.2f}%**")
