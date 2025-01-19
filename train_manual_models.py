import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse
import time


def main(config):
    start_time = time.time()  # Define the dataset folder
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
    y_scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "XGBoost": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=1000, max_depth=10, random_state=42
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(32, 16, 16, 16),
            activation="relu",
            max_iter=4000,
            random_state=42,
        ),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        "KNNReg": KNeighborsRegressor(n_neighbors=5),
    }

    # Train and evaluate models
    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predictions)
        results.append({"Model": name, "MSE": mse})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print results
    print("\nModel Comparison:")
    print(results_df)

    save_model_folder = "models/manual"
    os.makedirs(save_model_folder, exist_ok=True)

    for model_res in results:
        model_name = model_res["Model"]
        print(f"model name: {model_name}")
        model = models[model_name]
        save_path = os.path.join(
            save_model_folder, f"{model_name.replace(' ', '_').lower()}.joblib"
        )
        joblib.dump(model, save_path)

    time_taken = (time.time() - start_time) / 60
    print(f"creating dataset from {config.n_rows} rows took {time_taken:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=5000)
    config = parser.parse_args()
    main(config)
