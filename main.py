
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATASET
# ==================================================

dataset = load_breast_cancer()

df = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

df["target"] = dataset.target

df["diagnosis"] = df["target"].map({
    0: "Malignant",
    1: "Benign"
})


# ==================================================
# DATASET OVERVIEW
# ==================================================

print("=" * 60)
print("BREAST CANCER CLASSIFICATION SYSTEM")
print("=" * 60)

print("\nDATASET OVERVIEW")
print("-" * 45)

print("Total samples:", len(df))
print("Input features:", len(dataset.feature_names))
print("Target classes:", len(dataset.target_names))
print("Dataset shape:", df.shape)


# ==================================================
# FIRST FIVE RECORDS
# ==================================================

print("\nFIRST 5 RECORDS")
print("-" * 45)

print(
    df[
        [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "diagnosis"
        ]
    ].head().to_string(index=False)
)


# ==================================================
# DATA QUALITY CHECK
# ==================================================

print("\nDATA QUALITY CHECK")
print("-" * 45)

print("Missing values:", df.isnull().sum().sum())
print("Duplicate records:", df.duplicated().sum())


# ==================================================
# CLASS DISTRIBUTION
# ==================================================

class_distribution = (
    df["diagnosis"]
    .value_counts()
    .rename_axis("diagnosis")
    .reset_index(name="count")
)

class_distribution["percentage"] = (
    class_distribution["count"] / len(df) * 100
).round(2)

print("\nCLASS DISTRIBUTION")
print("-" * 45)

print(class_distribution.to_string(index=False))


# ==================================================
# DESCRIPTIVE STATISTICS
# ==================================================

feature_statistics = df[
    list(dataset.feature_names)
].describe().transpose()

print("\nFEATURE STATISTICS")
print("-" * 45)

print(
    feature_statistics[
        ["mean", "std", "min", "max"]
    ].head(10).round(4)
)


# ==================================================
# SAVE DATASET AND ANALYSIS
# ==================================================

df.to_csv(
    DATA_DIR / "breast_cancer.csv",
    index=False
)

class_distribution.to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False
)

feature_statistics.to_csv(
    OUTPUT_DIR / "feature_statistics.csv"
)

print("\nFILES GENERATED")
print("-" * 45)

print("data/breast_cancer.csv")
print("outputs/class_distribution.csv")
print("outputs/feature_statistics.csv")

print("\nStep 2 completed successfully!")
