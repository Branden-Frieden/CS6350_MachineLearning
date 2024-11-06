import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Scale the features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Train SVM
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Get probability estimates from both models
rf_probs = rf_model.predict_proba(X_val)
svm_probs = svm_model.predict_proba(X_val_scaled)

# Define a confidence threshold
threshold = 0.6

# Combine predictions based on confidence
final_predictions = []
for rf_prob, svm_prob in zip(rf_probs, svm_probs):
    rf_confidence = max(rf_prob)
    svm_confidence = max(svm_prob)
    
    if rf_confidence >= threshold:
        final_predictions.append(np.argmax(rf_prob))  # Use Random Forest prediction
    elif svm_confidence >= threshold:
        final_predictions.append(np.argmax(svm_prob))  # Use SVM prediction
    else:
        final_predictions.append(-1)  # -1 indicates abstain or unclassified

# Calculate accuracy
final_predictions = np.array(final_predictions)
valid_predictions = final_predictions[final_predictions != -1]  # Filter out abstains
valid_y_val = y_val[final_predictions != -1]  # Corresponding true values

accuracy = accuracy_score(valid_y_val, valid_predictions) * 100
print(f'Combined Model Accuracy: {accuracy:.2f}%')
