Student Grades Analysis

This project is part of a data preprocessing and modeling assignment using the Students Grading Dataset from Kaggle. It contains data cleaning, transformation, reduction, visualization, and modeling techniques to explore and predict student performance.

Files Included
- `student_grades_analysis.ipynb`: Jupyter Notebook, initial regression model
- `Student_grades_analysis_final_(1).ipynb`: Final Jupyter Notebook with classification model and full preprocessing
- `student_grades_analysis.py`: Python script version, core code only
- `Students_Grading_Dataset.csv`: Dataset used for modeling
- `total_score_histogram.png`: Histogram from regression phase
- `total_score_histogram_.png`: Histogram from classification phase

Methods Used
- Data Cleaning
  Removed duplicates: df.drop_duplicates(inplace=True)
  Handled missing values: df.dropna(inplace=True)
  Dropped irrelevant columns: Removed Student_ID, First_Name, Last_Name, and Email
  Converted categorical variables: pd.get_dummies() used for proper modeling
- Data Reduction
  Attribute Subset Selection
    Selected the most relevant features:
    ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Projects_Score', 'Study_Hours_per_Week']
    Removed other less important or redundant ones
  Multiple Linear Regression
    Trained on selected features
    Evaluated using:
    R² Score
    Mean Squared Error
  Histogram Analysis: Distribution of Total Scores
- Data Transformation
  Normalization was done using:
  scaler = MinMaxScaler()
  df[num_cols] = scaler.fit_transform(df[num_cols])
-Data Discretization
  Categorized total scores into "Low", "Medium", and "High"

Visualizations
- total_score_histogram.png: Histogram of Total Scores before classification
- total_score_histogram_.png: Histogram after discretization (Score_Level)

  
Modeling
- Multiple Linear Regression
- Evaluated using:
    R² Score
    Mean Squared Error

Random Forest Classifier
- Target: Score_Level (Low/Medium/High)
- Accuracy: ~87%
- Evaluation:
    Classification Report
    Confusion Matrix
    Feature Importance Plot

Results Summary
- Final Score and Study Hours were the most important predictors
- Classification model successfully categorized students into performance levels
