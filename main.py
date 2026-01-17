
# **Loan Approval Prediction System**

##**1. Data Loading (5 Marks)**


import pandas as pd
import numpy as np
import gradio as gr
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# Dataset load
df = pd.read_csv("loan_approval.csv")

# Display first few rows and shape
print(df.head())
print("Dataset shape:", df.shape)

"""## **2. Data Preprocessing (10 Marks)**
Perform and document at least 5 distinct preprocessing steps (e.g., handling missing values, encoding, scaling, outlier detection, feature engineering).
"""

# 1. Drop irrelevant columns
df = df.drop(['name','city'], axis=1)

# 2. Encode target variable
df['loan_approved'] = df['loan_approved'].astype(int)

# 3. Handle missing values
df.fillna(df.median(), inplace=True)

# 4. Outlier detection & capping
for col in ['loan_amount','points']:
    df[col] = np.where(df[col] > df[col].quantile(0.95),
                       df[col].quantile(0.95),
                       df[col])

# 5. Feature engineering
df['income_to_loan_ratio'] = df['income'] / (df['loan_amount']+1)

# 6. One-Hot Encoding (for categorical variables if present)
df = pd.get_dummies(df, drop_first=True)


# Train-Test Split
X = df.drop('loan_approved', axis=1)
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## **3. Pipeline Creation (10 Marks)**
Construct a standard Machine Learning pipeline that integrates preprocessing and the model
"""

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

"""## **4. Primary Model Selection (5 Marks)**
Choose a suitable algorithm and justify why this specific model was selected for the dataset.    
**Answer:**
We selected Logistic Regression as the primary model because it is well-suited for classification problems, provides interpretable results, and offers efficient training with regularization options. Its balance of simplicity and effectiveness makes it an appropriate choice for our dataset.

## **5. Model Training (10 Marks)**
Train your selected model using the training portion of your dataset.
"""

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

"""## **6. Cross-Validation (10 Marks)**
Apply Cross-Validation  to assess robustness and report the average score with standard deviation.

"""

# Apply 5-fold cross-validation on the training set
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

# Report average score and standard deviation
print("Cross-Validation Mean Accuracy:", scores.mean())
print("Cross-Validation Standard Deviation:", scores.std())

"""## **7. Hyperparameter Tuning (10 Marks)**
Optimize your model using search methods displaying both the parameters tested and the best results found.
"""

# Define parameter grid for Logistic Regression
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],          # Regularization strength
    'model__solver': ['liblinear', 'lbfgs']  # Optimization algorithms
}

# Apply GridSearchCV on the pipeline
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Display results
print("Parameters tested:", param_grid)
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Score:", grid.best_score_)

"""## **8. Best Model Selection (10 Marks)**
Select  the final best-performing model based on the hyperparameter tuning results.
"""

# Select the best model from GridSearchCV
best_model = grid.best_estimator_

# Display the chosen configuration
print("Final Best Model:", best_model)

"""## **9. Model Performance Evaluation (10 Marks)**
Evaluate the model on the test set and print comprehensive metrics suitable for the problem type.
"""

# Predict on the test set
y_pred = best_model.predict(X_test)

# Print evaluation metrics
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""## **Save & Load Model**"""

# joblib.dump(pipeline, "/content/drive/MyDrive/Resources/2-Phitron AIML/Machine Learning/Assignment & Exam/Module 29: ML Final Exam/model.pkl")
# joblib.dump(best_model, "/content/drive/MyDrive/Resources/2-Phitron AIML/Machine Learning/Assignment & Exam/Module 29: ML Final Exam/model.pkl")

import pickle

# Save the best model
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

"""**See rest of task on app.py file**"""