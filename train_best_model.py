import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse
import time

from predict import predict


def main(config):
    start_time = time.time()
    # Define the dataset folder
    dataset_folder = f"../dataset/{config.n_rows}"

    print("Loading dataset...")
    # Load training and testing data
    X_train = pd.read_csv(f"{dataset_folder}/X_train.csv")
    X_test = pd.read_csv(f"{dataset_folder}/X_test.csv")
    y_train = pd.read_csv(f"{dataset_folder}/y_train.csv")
    y_test = pd.read_csv(f"{dataset_folder}/y_test.csv")
    print("Dataset loaded.")

    print(f"X train shape: {X_train.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"X test shape: {X_test.shape}")
    print(f"y test shape: {y_test.shape}")

    # Standardize features
    scaler = StandardScaler()
    # y_scaler = StandardScaler()
    if config.normalize_data == "yes":
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Initialize models
    model_name = "XGBoost"
    model = XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=42,
    )  # Train and evaluate models
    results = []

    print(f"Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    results.append({"Model": model_name, "MSE": mse})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print results
    print("\nModel Comparison:")
    print(results_df)

    save_model_folder = "models/best"
    os.makedirs(save_model_folder, exist_ok=True)

    print(f"model name: {model_name}")
    save_path = os.path.join(
        save_model_folder, f"{model_name.replace(' ', '_').lower()}.joblib"
    )
    joblib.dump(model, save_path)

    _, mse_train, mse_test = predict(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        model_path=save_path,
        model_type="manual",
    )
    print(f"MSE test: {mse_test}\nMSE train: {mse_train}")
    time_taken = (time.time() - start_time) / 60
    print(f"creating dataset from {config.n_rows} rows took {time_taken:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=5000)
    parser.add_argument("--normalize_data", type=str, default="yes")
    parser.add_argument(
        "--n_estimators", type=int, default=600, help="Number of trees in the ensemble."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Step size shrinkage used to prevent overfitting.",
    )
    parser.add_argument(
        "--max_depth", type=int, default=10, help="Maximum depth of a tree."
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Fraction of samples to be used for each tree.",
    )
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.8,
        help="Fraction of features to be used for each tree.",
    )

    config = parser.parse_args()
    main(config)
