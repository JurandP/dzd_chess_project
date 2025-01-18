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

from feature_creation import get_features

# Check if embeddings should be written
write_embeddings = len(sys.argv) > 1 and sys.argv[1].lower() == 'write'
    
if len(sys.argv) > 2:
    nrows = int(sys.argv[2])
else:
    nrows = 1000


# Initialize chess engine
engine_path = "../stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

if write_embeddings:
    # Load the data
    print("loading_data ...")
    data = pd.read_csv("../lichess_db_puzzle.csv", nrows=nrows)
    print("loaded data :D")

    # Extract features
    X, y = get_features(df=data, chess_engine=engine)

    # Pickle X and y
    with open('local/X.pkl', 'wb') as f:
        pickle.dump(X, f)

    with open('local/y.pkl', 'wb') as f:
        pickle.dump(y, f)

    # Close the engine
    engine.quit()
else:
    print("Loading X and y from pickle files...")
    with open('local/X.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('local/y.pkl', 'rb') as f:
        y = pickle.load(f)
        
# Close the engine
engine.quit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X[:nrows], y[:nrows], test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaler = y_scaler.fit_transform(pd.DataFrame({'y': y_train}))
y_test_scaler = y_scaler.fit_transform(pd.DataFrame({'y': y_test}))

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
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
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

