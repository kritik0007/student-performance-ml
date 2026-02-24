import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "study_hours": [2, 4, 6, 8, 1, 3, 7, 5],
    "attendance": [60, 75, 80, 90, 50, 65, 85, 70],
    "previous_marks": [50, 55, 65, 80, 40, 60, 78, 68],
    "result": [0, 0, 1, 1, 0, 0, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nEnter Student Details:")
hours = float(input("Study Hours: "))
attendance = float(input("Attendance (%): "))
marks = float(input("Previous Marks: "))

prediction = model.predict([[hours, attendance, marks]])

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")