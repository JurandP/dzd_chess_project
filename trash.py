from tabnanny import verbose
import pandas as pd
import chess
import chess.engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.auto import tqdm
import joblib
import sys
import pickle


# Define the dataset folder
dataset_folder = "../dataset/"

print("Loading dataset...")
# Load training and testing data
X_train = pd.read_csv(f"{dataset_folder}X_train.csv")
X_test = pd.read_csv(f"{dataset_folder}X_test.csv")
y_train = pd.read_csv(f"{dataset_folder}y_train.csv")
y_test = pd.read_csv(f"{dataset_folder}y_test.csv")
print("Dataset loaded.")

print(f'X train shape: {X_train.shape}')
print(f'y train shape: {y_train.shape}')
print(f'X test shape: {X_test.shape}')
print(f'y test shape: {y_test.shape}')

# Standardize features
scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Initialize models
models = {
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(32, 16, 16, 16), activation='relu', max_iter=4000, random_state=42),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    "KNNReg": KNeighborsRegressor(n_neighbors=5)
}  

# Train and evaluate models
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train_scaled)
    predictions = model.predict(X_test_scaled)
    print(f'type(y_test): {type(y_test)}')
    print(f'y_test.shape: {y_test.shape}')
    print(f'type(predictions) before: {type(predictions)}')
    print(f'predictions.shape before: {predictions.shape}')
    if name != 'KNNReg':
        predictions = pd.DataFrame({'y_hat': predictions})
    print(f'type(predictions): {type(predictions)}')
    print(f'predictions.shape: {predictions.shape}')
    predictions = y_scaler.inverse_transform(predictions)
    mse = mean_squared_error(y_test, predictions)
    results.append({"Model": name, "MSE": mse})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results
print("\nModel Comparison:")
print(results_df)

# Save the best model
best_model_name = results_df.sort_values(by="MSE").iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")
best_model = models[best_model_name]
joblib.dump(best_model, f"local/{best_model_name.replace(' ', '_').lower()}_model.pkl")

# Save the scaler
joblib.dump(scaler, "local/scaler.pkl")
joblib.dump(y_scaler, "local/y_scaler.pkl")

