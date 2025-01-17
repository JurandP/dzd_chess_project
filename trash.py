import pandas as pd
import chess
import chess.engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.auto import tqdm
import joblib

from feature_creation import get_features, extract_features


# Load the data
print("loading_data ...")
data = pd.read_csv("../lichess_db_puzzle.csv", nrows=1000)
print("loaded data :D")

# Initialize chess engine204021.2
engine_path = "../stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

X, y = get_features(df=data, chess_engine=engine)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_test.head())

print("data prepared :D")
print("training model ...")

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("model trained :D")

# Evaluate the model
predictions = model.predict(X_test_scaled)
print(f"X_test:\n{X_test_scaled[:5]}")
print(f"y_test:\n{y_test.head()}")
print(f"predictions:\n{predictions[:5]}")
mse = mean_squared_error(y_test, predictions)
print(f" Mean Squared Error: {mse:.2f}")
sanity_mse = mean_squared_error(y_test, np.ones_like(y_test) * np.mean(y_train))
print(f" Sanity MSE: {sanity_mse:.2f}")

# Save the model
joblib.dump(model, "local/linear_chess_rating_model.pkl")
joblib.dump(scaler, "local/scaler.pkl")


# Example usage
example_fen = "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17"
example_moves = "e8d7 a2e6 d7d8 f7f8"

# Close the engine
engine.quit()
