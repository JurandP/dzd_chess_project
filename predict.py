from pycaret.regression import load_model, predict_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os
import argparse


def predict(X_train, y_train, X_test, y_test, model_path, model_type="automl"):
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    if model_type == "automl":
        # Load the PyCaret-tuned model
        model_path = model_path[:-4]
        model = load_model(model_path, verbose=False)

        # Make predictions on the train set
        predictions_train = predict_model(model, data=X_train)
        mse_train = mean_squared_error(y_train, predictions_train["prediction_label"])

        # Make predictions on the test set
        predictions_test = predict_model(model, data=X_test)
        mse_test = mean_squared_error(y_test, predictions_test["prediction_label"])

    elif model_type == "manual":

        model = joblib.load(model_path)

        # Make predictions on the train set
        predictions = model.predict(X_train)
        mse_train = mean_squared_error(y_train, predictions)

        # Make predictions on the test set
        predictions = model.predict(X_test)
        mse_test = mean_squared_error(y_test, predictions)

    return model_name, mse_train, mse_test


def main(config):
    # Define the dataset folder
    dataset_folder = f"../dataset/{config.n_rows}"

    print("Loading dataset...")
    # Load training and testing data
    X_train = pd.read_csv(f"{dataset_folder}X_train.csv")
    X_test = pd.read_csv(f"{dataset_folder}X_test.csv")
    y_train = pd.read_csv(f"{dataset_folder}y_train.csv")
    y_test = pd.read_csv(f"{dataset_folder}y_test.csv")
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

    results = {
        "model": [],
        "MSE-train": [],
        "MSE-test": [],
        "tuning mode": [],
    }

    for model_type in os.listdir("models"):
        if model_type in ['automl', 'manual']:
            print(model_type)
            model_dir = os.path.join("models", model_type)
            for model_file in os.listdir(model_dir):
                # print(f"model file: {model_file}")
                model_path = os.path.join(model_dir, model_file)
                if model_type == "automl":
                    model_name, mse_train, mse_test = predict(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        model_path=model_path,
                        model_type=model_type,
                    )
                else:
                    model_name, mse_train, mse_test = predict(
                        X_train_scaled,
                        y_train,
                        X_test_scaled,
                        y_test,
                        model_path=model_path,
                        model_type=model_type,
                    )
                print(f"{model_name}\t{mse_train:.0f}\t{mse_test:.0f}")
                results["model"].append(model_name)
                results["MSE-train"].append(mse_train)
                results["MSE-test"].append(mse_test)
                results["tuning mode"].append(model_type)

    results = pd.DataFrame(results)
    results = results.sort_values("MSE-test", ascending=True)
    print(results)
    results.to_csv("model_search.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=5000)
    config = parser.parse_args()
    main(config)
