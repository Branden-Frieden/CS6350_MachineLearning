from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import pandas as pd

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

for i in range(12):
    # Using Random Forest as a base model for RFE
    rf_model = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=rf_model, n_features_to_select=i+1)
    rfe.fit(X_train, y_train)

    # Get selected features
    selected_features = df.columns[:][rfe.support_]
    print("Selected Features:", selected_features.tolist())

    # Create new DataFrames based on selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    # Train a Random Forest model using the selected features
    rf_model_selected = RandomForestClassifier(random_state=42)
    rf_model_selected.fit(X_train_selected, y_train)

    # Make predictions on the validation set
    y_val_pred = rf_model_selected.predict(X_val_selected)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy with Selected Features: {accuracy * 100:.2f}%")
