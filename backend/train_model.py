"""
One-time training script.

Usage:
    python train_model.py path/to/FARM_NDVI.xlsx

Trains the RandomForest crop classifier on your historical NDVI dataset
(same logic as the notebook) and saves it to crop_model.pkl, which app.py
loads at server startup.
"""

import sys
from model import train_model


def main():
    if len(sys.argv) != 2:
        print("Usage: python train_model.py path/to/FARM_NDVI.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    print(f"Training model from {excel_path} ...")

    rf, feature_df = train_model(excel_path)

    print(f"Trained on {len(feature_df)} lifecycles across {feature_df['crop'].nunique()} crops.")
    print(feature_df["crop"].value_counts())
    print("\nSaved model to crop_model.pkl")


if __name__ == "__main__":
    main()
