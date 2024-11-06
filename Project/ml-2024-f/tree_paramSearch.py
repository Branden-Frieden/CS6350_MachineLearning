import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

# Load the data
df = pd.read_csv("train_final.csv")

# Convert categorical columns to numerical codes
for col in df.columns[:-1]:  # Exclude the target column
    df[col] = df[col].astype('category').cat.codes

# Get the answer column from train data
answers = df.iloc[:, -1].copy()
df = df.drop(columns=df.columns[-1])

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df, 
    answers, 
    test_size=0.2, 
    random_state=42
)

# Define the model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [150, 175, 200, 225, 250],
    'max_depth': [None, 15, 20, 25],
    'min_samples_split': [8, 10, 12, 14, 16, 18],
    'min_samples_leaf': [1, 2, 3]
}

# Set up the grid search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
