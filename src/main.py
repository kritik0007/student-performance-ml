import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/data.csv")

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

import matplotlib.pyplot as plt

# Correlation matrix
print("\nCorrelation Matrix:\n")
print(df.corr())

# Scatter plot
plt.figure()
plt.scatter(df["study_hours"], df["previous_marks"])
plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.title("Study Hours vs Previous Marks")
plt.show()

# Attendance vs Result
plt.figure()
plt.scatter(df["attendance"], df["result"])
plt.xlabel("Attendance")
plt.ylabel("Result (0=Fail, 1=Pass)")
plt.title("Attendance vs Result")
plt.show()