Student Grades Analysis

This project is part of a data preprocessing and modeling assignment using the Students Grading Dataset from Kaggle.

Files Included
- `student_grades_analysis.ipynb`: Jupyter Notebook version
- `student_grades_analysis.py`: Python script version
- `Students_Grading_Dataset.csv`: Dataset used for modeling
- `total_score_histogram.png`: Output histogram

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
- A histogram showing the distribution of Total Scores

  
Output
- R² Score and Mean Squared Error printed to evaluate model
- Histogram showing distribution of total scores
