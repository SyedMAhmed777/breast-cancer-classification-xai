
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "breast_cancer.csv"


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(DATA_FILE)

features = df.drop(
    columns=["target", "diagnosis"]
)

# Visualization styling
sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "sans-serif"

COLORS = {
    "Benign": "#3B82F6",
    "Malignant": "#F97316"
}


def save_figure(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Generated: {filename}")


print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ==================================================
# 1. CLASS DISTRIBUTION
# ==================================================

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="diagnosis",
    hue="diagnosis",
    order=["Benign", "Malignant"],
    palette=COLORS,
    legend=False
)

for container in ax.containers:
    ax.bar_label(container, padding=5)

plt.title(
    "Diagnosis Class Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")
plt.ylim(0, 400)

save_figure("class_distribution.png")


# ==================================================
# 2. FEATURE CORRELATION HEATMAP
# ==================================================

correlation = features.corr()

plt.figure(figsize=(18, 15))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.2,
    cbar_kws={"shrink": 0.8}
)

plt.title(
    "Feature Correlation Heatmap",
    fontsize=18,
    fontweight="bold"
)

save_figure("correlation_heatmap.png")


# ==================================================
# 3. FEATURE DISTRIBUTIONS
# ==================================================

selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean concavity",
    "worst radius"
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

for feature, ax in zip(
    selected_features,
    axes.flatten()
):

    sns.histplot(
        data=df,
        x=feature,
        hue="diagnosis",
        palette=COLORS,
        bins=25,
        kde=True,
        alpha=0.5,
        common_norm=False,
        stat="density",
        ax=ax
    )

    ax.set_title(
        feature.title(),
        fontweight="bold"
    )

fig.suptitle(
    "Feature Distributions by Diagnosis",
    fontsize=18,
    fontweight="bold"
)

save_figure("feature_distributions.png")


# ==================================================
# 4. FEATURE BOX PLOTS
# ==================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

for feature, ax in zip(
    selected_features,
    axes.flatten()
):

    sns.boxplot(
        data=df,
        x="diagnosis",
        y=feature,
        hue="diagnosis",
        order=["Benign", "Malignant"],
        palette=COLORS,
        legend=False,
        ax=ax
    )

    ax.set_title(
        feature.title(),
        fontweight="bold"
    )

    ax.set_xlabel("")

fig.suptitle(
    "Feature Comparison by Diagnosis",
    fontsize=18,
    fontweight="bold"
)

save_figure("feature_boxplots.png")


# ==================================================
# 5. SCATTER PLOT
# ==================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="mean radius",
    y="mean texture",
    hue="diagnosis",
    palette=COLORS,
    alpha=0.75,
    s=65
)

plt.title(
    "Mean Radius vs Mean Texture",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")

save_figure("feature_scatterplot.png")


# ==================================================
# 6. FEATURE CORRELATION WITH MALIGNANCY
# ==================================================

# Original dataset:
# 0 = Malignant
# 1 = Benign
#
# Create a temporary indicator:
# 1 = Malignant
# 0 = Benign

malignant_indicator = (
    df["target"] == 0
).astype(int)

target_correlation = features.corrwith(
    malignant_indicator
)

correlation_report = pd.DataFrame({
    "feature": target_correlation.index,
    "correlation": target_correlation.values
})

correlation_report["absolute_correlation"] = (
    correlation_report["correlation"].abs()
)

correlation_report = (
    correlation_report.sort_values(
        "absolute_correlation",
        ascending=False
    )
)

top_features = correlation_report.head(15)

plt.figure(figsize=(11, 8))

sns.barplot(
    data=top_features,
    x="absolute_correlation",
    y="feature",
    color="#3B82F6"
)

plt.title(
    "Top 15 Features Associated with Malignancy",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Absolute Pearson Correlation")
plt.ylabel("Feature")
plt.xlim(0, 1)

save_figure("target_correlation.png")


# ==================================================
# SAVE ANALYSIS REPORT
# ==================================================

correlation_report.to_csv(
    OUTPUT_DIR / "target_correlation.csv",
    index=False
)

print("\nTOP 10 FEATURES BY ABSOLUTE CORRELATION")
print("-" * 60)

print(
    correlation_report.head(10)
    .round(4)
    .to_string(index=False)
)

print("\nAll visualizations saved to:")
print(FIGURE_DIR)

print("\nStep 3 completed successfully!")
