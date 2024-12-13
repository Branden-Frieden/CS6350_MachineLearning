import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Load the data
df = pd.read_csv("train_final.csv")

test_df = pd.read_csv("test_final.csv")

random_state = 42

# Convert categorical columns to numerical codes
for col in df.columns[:-1]:  # Exclude the target column
    df[col] = df[col].astype('category').cat.codes
    test_df[col] = test_df[col].astype('category').cat.codes


# Get the answer column from train data
answers = df.iloc[:, -1].copy()
df = df.drop(columns=df.columns[-1])
df = df.drop(columns=["native.country"])


Ids = test_df.iloc[:, 0].copy()
test_df = test_df.drop(columns=test_df.columns[0])
test_df = test_df.drop(columns=["native.country"])


# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df, 
    answers, 
    test_size=0.2, 
    random_state=42
)

# Scale the features for SVM, Logistic Regression, KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.fit_transform(test_df)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
rf_model.fit(X_train, y_train)

# Train SVM
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Train Logistic Regression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Train K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)


# Get probability estimates from all models
rf_probs = rf_model.predict_proba(X_val)
svm_probs = svm_model.predict_proba(X_val_scaled)
logistic_probs = logistic_model.predict_proba(X_val_scaled)
gb_probs = gb_model.predict_proba(X_val)
knn_probs = knn_model.predict_proba(X_val_scaled)  # KNN requires scaled data

# Combine predictions based on confidence
final_predictions = []
final_preds = []
for rf_prob, svm_prob, log_prob, gb_prob, knn_prob in zip(rf_probs, svm_probs, logistic_probs, gb_probs, knn_probs):
    confidences = []
    probs = []

    # Append probabilities and confidences from each model
    probs.append(rf_prob)
    confidences.append(max(rf_prob))

    # probs.append(svm_prob)
    confidences.append(max(svm_prob))

    # probs.append(log_prob)
    confidences.append(max(log_prob))

    # probs.append(gb_prob)
    confidences.append(max(gb_prob))

    # probs.append(knn_prob)
    confidences.append(max(knn_prob))

    first_elements = [arr[0] for arr in probs]

    mean = np.mean(first_elements)

    if mean >= .5:
        final_predictions.append(0)
        final_preds.append(1 - np.mean(first_elements))
    else: 
        final_predictions.append(1)
        final_preds.append(1 - np.mean(first_elements))


# Calculate accuracy, excluding any abstains (-1) if you have implemented abstains
final_predictions = np.array(final_predictions)

# for i, pred in enumerate(final_predictions):

#     if pred != y_val.iloc[i]:
#         print(y_val.iloc[i])
#         print(final_preds[i], "\n")
    

accuracy = accuracy_score(y_val, final_predictions) * 100
print(f'Combined Model Accuracy: {accuracy:.2f}%')






rf_probs = rf_model.predict_proba(test_df)
svm_probs = svm_model.predict_proba(X_test_scaled)
logistic_probs = logistic_model.predict_proba(X_test_scaled)
gb_probs = gb_model.predict_proba(test_df)
knn_probs = knn_model.predict_proba(X_test_scaled)

final_predictions = []
final_preds = []


for rf_prob, svm_prob, log_prob, gb_prob, knn_prob in zip(rf_probs, svm_probs, logistic_probs, gb_probs, knn_probs):
    confidences = []
    probs = []

    # Append probabilities and confidences from each model
    probs.append(rf_prob)
    confidences.append(max(rf_prob))

    # probs.append(svm_prob)
    confidences.append(max(svm_prob))

    # probs.append(log_prob)
    confidences.append(max(log_prob))

    # probs.append(gb_prob)
    confidences.append(max(gb_prob))

    # probs.append(knn_prob)
    confidences.append(max(knn_prob))

    first_elements = [arr[0] for arr in probs]

    mean = np.mean(first_elements)


    if mean >= .5:
        final_predictions.append(0)
        final_preds.append(1 - np.mean(first_elements))
    else: 
        final_predictions.append(1)
        final_preds.append(1 - np.mean(first_elements))

        



with open("predictions.csv", "w") as file:
    # Write the header
    file.write("ID, Prediction\n")
    
    # Write each prediction to the file
    for i, pred in enumerate(final_preds):
        file.write(f"{i + 1}, {sigmoid(pred)}\n")