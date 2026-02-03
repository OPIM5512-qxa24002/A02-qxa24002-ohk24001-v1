# A02 – MLP Regression on California Housing Data

## Project Purpose
The purpose of this assignment is to build a simple end-to-end machine learning pipeline using the California Housing dataset. The project demonstrates data loading, model training with early stopping, and evaluation using train and test predictions.

## Workflow
1. Load the California Housing dataset from `sklearn`
2. Split the data into training and test sets
3. Train an `MLPRegressor` with `early_stopping=True` and custom hyperparameters
4. Generate predictions for both train and test data
5. Create and save Actual vs. Predicted plots for model evaluation

## Project Structure

src/
├─ ds_pipeline.py
└─ figs/
├─ train_actual_vs_pred.png
└─ test_actual_vs_pred.png


The `figs` folder is created automatically when the script is run.

## How to Run
From the repository root:
pip install -r requirements.txt
python src/ds_pipeline.py

Team Members

    Joseph Nartey
    Niharika Sharma
