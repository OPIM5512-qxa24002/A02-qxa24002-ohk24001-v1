# src/ds_pipeline.py

def main():
    # ============================================================
    # PR #1: Load dataset + train/test split
    # ============================================================
    # (Imports introduced in PR #1)
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    # Load California Housing dataset from sklearn
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # includes features + target

    # Split into X (features) and y (target)
    target_col = "MedHouseVal"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )