
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data/breast_cancer.csv")

# 2. Separate inputs and target
X = df.drop(columns=["target", "diagnosis"])
y = df["target"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the model with feature scaling
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Malignant", "Benign"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 8. Save the trained model
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "logistic_model.joblib")

print("\nModel saved successfully!")
