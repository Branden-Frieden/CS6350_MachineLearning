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

# Load the data
df = pd.read_csv("test_final.csv")

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

# Scale the features for SVM, Logistic Regression, KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
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

    probs.append(svm_prob)
    confidences.append(max(svm_prob))

    probs.append(log_prob)
    confidences.append(max(log_prob))

    probs.append(gb_prob)
    confidences.append(max(gb_prob))

    probs.append(knn_prob)
    confidences.append(max(knn_prob))

    first_elements = [arr[0] for arr in probs]


    if np.mean(first_elements) >= .5:
        final_predictions.append(0)
        final_preds.append(np.mean(first_elements))
    else: 
        final_predictions.append(1)
        final_preds.append(np.mean(first_elements))


# Calculate accuracy, excluding any abstains (-1) if you have implemented abstains
final_predictions = np.array(final_predictions)

for i, pred in enumerate(final_predictions):

    if pred != y_val.iloc[i]:
        print(y_val.iloc[i])
        print(final_preds[i], "\n")
    

accuracy = accuracy_score(y_val, final_predictions) * 100
print(f'Combined Model Accuracy: {accuracy:.2f}%')



