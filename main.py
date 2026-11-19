# 1. Import pandas library
import pandas as pd

# 2. Read the student CSV file
data = pd.read_csv("student.csv")

# 3. Display first 5 student records
print("First 5 Student Records:")
print(data.head())

# 4. Display rows and columns
print("\nDataset Shape:")
print(data.shape)

# 5. Display dataset information
print("\nDataset Information:")
data.info()
# Step 6: Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 7: Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Step 8: Remove duplicate records
data = data.drop_duplicates()

# Step 9: Separate input and output
X = data[['study_hours', 'attendance', 'previous_marks']]
y = data['final_marks']

print("\nInput Data:")
print(X.head())

print("\nOutput Data:")
print(y.head())
# Step 10: Split data into training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Size:")
print(X_train.shape)

print("\nTesting Data Size:")
print(X_test.shape)
# Step 12: Create Linear Regression Model

from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")
# Step 13: Predict Student Marks

# New student details
new_student = pd.DataFrame(
    [[5, 80, 75]],
    columns=X.columns
)

# Predict final marks
prediction = model.predict(new_student)

print("\nStudent Marks Prediction:")
print("Predicted Final Marks:", prediction[0])
# Step 14: Evaluate Model Performance

from sklearn.metrics import mean_absolute_error, r2_score

# Predict test data
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
# Predict test data
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
# Step 15: Actual vs Predicted Graph

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

# Actual marks
plt.scatter(y_test, y_pred)

# Graph labels
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")

# Show graph
plt.show()
# Actual vs Predicted Graph

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred, label="Predicted Marks")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    label="Perfect Prediction"
)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")

plt.legend()
plt.grid()

plt.show()
metrics = ["MAE", "R2 Score"]
values = [mae, r2]

plt.figure(figsize=(7, 5))
plt.bar(metrics, values)
plt.title("Model Performance")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.show()