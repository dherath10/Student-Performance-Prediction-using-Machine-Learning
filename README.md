Lab Sheet: Student Performance Prediction using Machine Learning

Title: Student Performance Prediction using Feature Engineering and Multiple Regression Models
Duration: 2 hours
Tools: Python, Scikit-learn, XGBoost, LightGBM, Jupyter Notebook
Objective
By the end of this lab, students will be able to:
- Preprocess raw educational data (handling missing values, encoding, scaling).
- Train multiple regression models (MLP, XGBoost, LightGBM, Random Forest, SVM).
- Evaluate and compare model performance using MSE, RMSE, and R².
Dataset
Mendeley Data  - Students_Performance_data_set.xlsx
A dataset containing student demographics, behavioral patterns, and current CGPA.
Instructions
Follow the steps below in your Jupyter Notebook:
•	Step 1: Import Required Libraries

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

•	Step 2: Load and Inspect Dataset

df = pd.read_excel("Students_Performance_data_set.xlsx")
df.head()

•	Step 3: Data Preprocessing (Handle missing values, encode, normalize)
•	Identify and separate numerical and categorical columns.
•	Handle missing values.
•	Apply label encoding and one-hot encoding.
•	Use MinMaxScaler to normalize the data.

# Identify target and feature
target_col = 'What is your current CGPA?'
one_hot_col = 'Status of your English language proficiency'

# Separate column types
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols.remove(target_col)
categorical_cols.remove(one_hot_col)

# Impute missing values
df[numerical_cols] = SimpleImputer(strategy='median').fit_transform(df[numerical_cols])
df[categorical_cols + [one_hot_col]] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols + [one_hot_col]])

# Encode categorical data
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df = pd.get_dummies(df, columns=[one_hot_col], prefix="English")

# Normalize
scaler = MinMaxScaler()
df[numerical_cols + [target_col]] = scaler.fit_transform(df[numerical_cols + [target_col]])

•	Step 4: Train-Test Split
•	70% of the data is assigned to the training set (X_train, y_train).
•	The remaining 30% is stored in a temporary set (X_temp, y_temp) for further splitting.
•	The random_state parameter acts as a seed for the random number generator.

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

•	Step 5: Model Training and Evaluation using XGBoost, LightGBM, MLP, Random Forest, SVM

Evaluation Function
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(name, y_true, y_pred):
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)print
(f"{name} => MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
return {"Model": name, "MSE": mse, "RMSE": rmse, "R2": r2}

Train Models
Each model has strengths:
•	XGBoost & LightGBM: Fast and accurate for structured data.
•	Random Forest: Robust and interpretable.
•	MLP: Captures non-linear relationships.
•	SVR: Effective for smooth and small datasets.


from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

results = []

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
xgb_model.fit(X_train, y_train)
results.append(evaluate_model("XGBoost", y_test, xgb_model.predict(X_test)))

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05)
lgb_model.fit(X_train, y_train)
results.append(evaluate_model("LightGBM", y_test, lgb_model.predict(X_test)))

# MLP
mlp_model = MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000)
mlp_model.fit(X_train, y_train)
results.append(evaluate_model("MLP", y_test, mlp_model.predict(X_test)))

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
results.append(evaluate_model("Random Forest", y_test, rf_model.predict(X_test)))

# SVM
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
results.append(evaluate_model("SVM", y_test, svm_model.predict(X_test)))

•	Step 6: Summary Table of Results

# Summary Table
pd.DataFrame(results)

References
•	Scikit-learn documentation: https://scikit-learn.org/stable/
•	XGBoost documentation: https://xgboost.readthedocs.io/
•	LightGBM documentation: https://lightgbm.readthedocs.io/
