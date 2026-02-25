import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/data.csv")

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("🎓 Student Performance Predictor")

st.write("Enter student details below:")

study_hours = st.slider("Study Hours", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_marks = st.slider("Previous Marks", 0, 100, 60)

if st.button("Predict Result"):
    prediction = model.predict([[study_hours, attendance, previous_marks]])

    if prediction[0] == 1:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")