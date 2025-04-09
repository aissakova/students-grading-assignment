# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Students_Grading_Dataset.csv")

# ----------------- Data Cleaning -----------------
# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Drop irrelevant columns
df.drop(columns=["Student_ID", "First_Name", "Last_Name", "Email"], inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# ----------------- Data Transformation -----------------
# Normalize numeric columns
scaler = MinMaxScaler()
num_cols = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score',
            'Projects_Score', 'Total_Score', 'Study_Hours_per_Week',
            'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
df[num_cols] = scaler.fit_transform(df[num_cols])

# ----------------- Data Reduction -----------------
# Attribute subset selection
features = ['Attendance (%)', 'Midterm_Score', 'Final_Score',
            'Projects_Score', 'Study_Hours_per_Week']
target = 'Total_Score'

X = df[features]
y = df[target]

# ----------------- Modeling -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# ----------------- Histogram Analysis -----------------
plt.figure(figsize=(10, 6))
sns.histplot(df['Total_Score'], kde=True)
plt.title('Distribution of Total Scores')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("total_score_histogram.png")
plt.show()