import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/data.csv")

print("\nFirst 5 Rows of Dataset:\n")
print(df.head())


# ==============================
# Feature Selection
# ==============================

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]


# ==============================
# Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# Model Training
# ==============================

model = LogisticRegression()
model.fit(X_train, y_train)


# ==============================
# Model Evaluation
# ==============================

predictions = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ==============================
# Exploratory Data Analysis
# ==============================

print("\nCorrelation Matrix:\n")
print(df.corr())


# Scatter Plot 1
plt.figure()
plt.scatter(df["study_hours"], df["previous_marks"])
plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.title("Study Hours vs Previous Marks")
plt.show()


# Scatter Plot 2
plt.figure()
plt.scatter(df["attendance"], df["result"])
plt.xlabel("Attendance")
plt.ylabel("Result (0=Fail, 1=Pass)")
plt.title("Attendance vs Result")
plt.show()