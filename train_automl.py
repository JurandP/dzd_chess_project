import pandas as pd
from pycaret.regression import (
    setup,
    compare_models,
    save_model,
    tune_model,
    predict_model,
)
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import os
import argparse
import time


def main(config):
    start_time = time.time()  # Define the dataset folder
    # Define the dataset folder
    dataset_folder = f"../dataset/{config.n_rows}"

    print("Loading dataset...")
    # Load training and testing data
    X_train = pd.read_csv(f"{dataset_folder}/X_train.csv")
    X_test = pd.read_csv(f"{dataset_folder}/X_test.csv")
    y_train = pd.read_csv(f"{dataset_folder}/y_train.csv")
    y_test = pd.read_csv(f"{dataset_folder}/y_test.csv")
    print("Dataset loaded.")

    # Ensure that y_train and y_test are Series (1D) rather than DataFrames
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    # Combine X_train and y_train into a single DataFrame for PyCaret
    train_data = X_train.copy()
    train_data[
        "target"
    ] = y_train  # Rename 'target' to your actual target column name if different

    # Initialize the PyCaret setup
    reg_setup = setup(
        data=train_data,
        target="target",  # Replace 'target' with your actual target column name
        session_id=123,
        verbose=False,
    )

    # Define models to train
    models_to_train = [
        "lr",  # Linear Regression
        "lasso",  # Lasso Regression
        "ridge",  # Ridge Regression
        "dt",  # Decision Tree
        "rf",  # Random Forest
        "et",  # Extra Trees
        "gbr",  # Gradient Boosting
        "ada",  # AdaBoost
    ]

    print("Training and comparing models...")

    # Compare models and select top 3
    top3_models = compare_models(include=models_to_train, n_select=3)

    tuned_models = []
    print("Tuning top 3 models...")
    # Tune each of the top 3 models
    for model in tqdm(top3_models, desc="tuning models"):
        tuned = tune_model(model, verbose=False)
        tuned_models.append(tuned)

    print("Evaluating tuned models...")

    save_model_folder = "models/automl"
    os.makedirs(save_model_folder, exist_ok=True)

    # Evaluate each tuned model by calculating MSE on the test set
    for i, model in enumerate(tuned_models, start=1):
        # Predict on the test set
        predictions = predict_model(model, data=X_test)
        # Calculate MSE
        # print('predictions:')
        # print(type(predictions))
        # print(predictions.keys())
        mse = mean_squared_error(y_test, predictions["prediction_label"])
        print(f"MSE for: ({model.__class__.__name__}):\t{mse:.0f}")

        # (Optional) Save the tuned model to disk
        save_path = os.path.join(save_model_folder, model.__class__.__name__)
        save_model(model, save_path)

    print("Top 3 tuned models have been evaluated and saved.")
    time_taken = (time.time() - start_time) / 60
    print(f"creating dataset from {config.n_rows} rows took {time_taken:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=5000)
    config = parser.parse_args()
    main(config)
