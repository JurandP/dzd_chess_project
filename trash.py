import pandas as pd
import chess
import chess.engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm.auto import tqdm

from feature_creation import get_chess_engine_features, extract_features

# Load the CSV data
print('loading_data ...')
data = pd.read_csv('../lichess_db_puzzle.csv', nrows=1000)
print('loaded data :D')

# Initialize chess engine (adjust the path to your Stockfish binary)
engine_path = "../stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

X, y = get_chess_engine_features(df=data.copy(), chess_engine=engine)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('data prepared :D')
print('training model ...')

# Train a linear classifier
model = LinearRegression()
model.fit(X_train, y_train)

print('model trained :D')

# Predict and evaluate
predictions = model.predict(X_test)
print(f'X_test:\n{X_test.head()}')
print(f'y_test:\n{y_test.head()}')
print(f'predictions:\n{predictions[:5]}')
rmse = mean_squared_error(y_test, predictions)
print(f"Root Mean Squared Error: {rmse:.2f}")

# Save the model (optional)
import joblib
joblib.dump(model, 'linear_chess_rating_model.pkl')

# Example usage
def predict_rating(fen, moves):
    features = extract_features(fen, moves, chess_engine=engine)
    return model.predict([features])[0]

# Example prediction
example_fen = "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17"
example_moves = "e8d7 a2e6 d7d8 f7f8"
print(f"Predicted Rating: {predict_rating(example_fen, example_moves)}")

# Close the engine
engine.quit()