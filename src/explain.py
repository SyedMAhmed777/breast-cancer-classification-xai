
from pathlib import Path

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 1. Load dataset and trained model
BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data/breast_cancer.csv")
model = joblib.load(BASE_DIR / "models/logistic_model.joblib")

X = df.drop(columns=["target", "diagnosis"])
y = df["target"]

# 2. Recreate the original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create the SHAP explainer
explainer = shap.Explainer(
    model.predict_proba,
    X_train
)

# 4. Explain 20 testing samples
samples = X_test.head(20)
shap_values = explainer(samples)

# Select malignant class (class 0)
explanations = shap_values[:, :, 0]

# 5. Create output folder
output_dir = BASE_DIR / "outputs" / "shap"
output_dir.mkdir(parents=True, exist_ok=True)

# 6. Explain one prediction
shap.plots.waterfall(explanations[0], show=False)
plt.savefig(
    output_dir / "individual_prediction.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 7. Show overall feature importance
shap.plots.bar(explanations, show=False)
plt.savefig(
    output_dir / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("SHAP analysis completed successfully!")
print("Samples explained:", len(samples))
print("Graphs saved to:", output_dir)
