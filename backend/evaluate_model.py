"""
Model evaluation script — precision, recall, F1, and cross-validated accuracy.

This mirrors the evaluation done in the notebook:
  - 70/30 stratified train/test split
  - classification_report (precision, recall, f1-score per crop)
  - confusion matrix (saved as an image, since this runs outside a notebook)
  - 5-fold stratified cross-validation accuracy

This does NOT save a model for the website to use — it's purely to check how
good the model is. Use train_model.py separately to train and save the
model the backend actually serves.

Usage:
    python evaluate_model.py path/to/FARM_NDVI.xlsx
"""

import sys

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import build_training_dataset
from lifecycle import FEATURE_COLUMNS

N_ESTIMATORS = 50
RANDOM_STATE = 42
TEST_SIZE = 0.3


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_model.py path/to/FARM_NDVI.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    print(f"Building training dataset from {excel_path} ...")
    feature_df = build_training_dataset(excel_path)

    if feature_df.empty:
        print("No valid lifecycles were extracted — check the Excel file's format.")
        sys.exit(1)

    print(f"Total lifecycles: {len(feature_df)}")
    print(feature_df["crop"].value_counts())
    print()

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["crop"]

    # ---- 70/30 held-out test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print("===== HELD-OUT TEST SET (30% of data) =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print()
    print("Precision / Recall / F1 per crop:")
    print(classification_report(y_test, y_pred))

    # ---- confusion matrix, saved as an image ----
    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=rf.classes_, yticklabels=rf.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion matrix image to confusion_matrix.png")
    print()

    # ---- 5-fold cross-validation accuracy (uses the full dataset) ----
    rf_cv = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(rf_cv, X, y, cv=cv, scoring="accuracy")

    print("===== 5-FOLD CROSS-VALIDATION (accuracy) =====")
    print("Fold scores:", scores)
    print("Mean accuracy:", scores.mean())
    print("Std:", scores.std())


if __name__ == "__main__":
    main()
