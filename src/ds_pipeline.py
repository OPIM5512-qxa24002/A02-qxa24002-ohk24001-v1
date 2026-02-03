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
        # ============================================================
    # PR #2: Add MLPRegressor with early stopping
    # ============================================================
    # (Imports introduced in PR #2)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
 
    # Train an MLPRegressor with early_stopping=True
    # + add at least one custom hyperparameter
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            early_stopping=True,      # REQUIRED
            random_state=42,
 
            # Custom hyperparameters (example set)
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            learning_rate_init=1e-3,
 
            max_iter=2000
        ))
    ])
 
    model.fit(X_train, y_train)

    # ============================================================
    # PR #3: Add train predictions + plot
    # ============================================================
    # (Imports introduced in PR #3)
    import numpy as np
    import matplotlib.pyplot as plt

    # Train predictions
    y_train_pred = model.predict(X_train)

    # Helper to build the plot (introduced in PR #3)
    def plot_actual_vs_pred(y_true, y_pred, title):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        plt.figure(figsize=(7, 6))
        plt.scatter(y_true, y_pred, alpha=0.4)

        # y=x line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val])

        plt.title(title)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()

    # Create the train plot object (saving happens later after PR #5)
    plot_actual_vs_pred(
        y_true=y_train,
        y_pred=y_train_pred,
        title="Actual vs Predicted (Train)"
    )

    # Store the current figure for later saving
    train_fig = plt.gcf()
    plt.close()

     # ============================================================
    # PR #4: Add test predictions + plot
    # ============================================================
    # Test predictions
    y_test_pred = model.predict(X_test)

    # Create the test plot object (saving happens later after PR #5)
    plot_actual_vs_pred(
        y_true=y_test,
        y_pred=y_test_pred,
        title="Actual vs Predicted (Test)"
    )

    # Store the current figure for later saving
    test_fig = plt.gcf()
    plt.close()

     # ============================================================
    # PR #5: Improve plots/labels/titles/metrics + Save files
    # ============================================================
    # (Imports introduced in PR #5)
    import os
    from sklearn.metrics import mean_squared_error, r2_score

    # Metrics (sklearn-version-safe RMSE)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = train_mse ** 0.5
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = test_mse ** 0.5
    test_r2 = r2_score(y_test, y_test_pred)

    # Update titles to include metrics (polish)
    # Rebuild plots with improved titles, then save
    # ----------------------------
    # Saving the files
    # ----------------------------
    figs_dir = os.path.join("src", "figs")
    os.makedirs(figs_dir, exist_ok=True)

    train_plot_path = os.path.join(figs_dir, "train_actual_vs_pred.png")
    test_plot_path = os.path.join(figs_dir, "test_actual_vs_pred.png")

    # Recreate train plot with metrics title and save
    plot_actual_vs_pred(
        y_true=y_train,
        y_pred=y_train_pred,
        title=f"Actual vs Predicted (Train)\nRMSE={train_rmse:.3f} | R²={train_r2:.3f}"
    )
    plt.savefig(train_plot_path, dpi=200)
    plt.close()

    # Recreate test plot with metrics title and save
    plot_actual_vs_pred(
        y_true=y_test,
        y_pred=y_test_pred,
        title=f"Actual vs Predicted (Test)\nRMSE={test_rmse:.3f} | R²={test_r2:.3f}"
    )
    plt.savefig(test_plot_path, dpi=200)
    plt.close()

    print("Saved plots:")
    print(f"- {train_plot_path}")
    print(f"- {test_plot_path}")
    print("\nPerformance:")
    print(f"Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f}, Test  R²: {test_r2:.4f}")


if __name__ == "__main__":
    main()

     