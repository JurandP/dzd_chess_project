import pandas as pd
import chess
import chess.engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm.auto import tqdm

# Load the CSV data
print('loading_data ...')
data = pd.read_csv('../lichess_db_puzzle.csv', nrows=1000)
print('loaded data :D')

# Initialize chess engine (adjust the path to your Stockfish binary)
engine_path = "../stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Function to extract features from FEN and moves
def extract_features(fen, moves):
    board = chess.Board(fen)
    evals = []
    for move in moves.split():
        try:
            move = chess.Move.from_uci(move)
            if move in board.legal_moves:
                board.push(move)
                eval_info = engine.analyse(board, chess.engine.Limit(depth=5))
                evals.append(eval_info['score'].relative.score(mate_score=10000))
            else:
                break
        except Exception as e:
            print(f"Error processing move {move}: {e}")
            break
    # Use mean and std of evaluations as features
    if evals:
        return [np.mean(evals), np.std(evals)]
    return [0, 0]

# Process the dataset to add features
tqdm.pandas()
data['features'] = data.progress_apply(lambda row: extract_features(row['FEN'], row['Moves']), axis=1)
print('features extracted :D')
data[['mean_eval', 'std_eval']] = pd.DataFrame(data['features'].tolist(), index=data.index)
data = data.drop(columns=['features'])

# Prepare training and test data
X = data[['mean_eval', 'std_eval']]
y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('data prepared :D')
print('training model ...')

# Train a linear classifier
model = LinearRegression()
model.fit(X_train, y_train)

print('model trained :D')

# Predict and evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse}")

# Save the model (optional)
import joblib
joblib.dump(model, 'linear_chess_rating_model.pkl')

# Close the engine
engine.quit()

# Example usage
def predict_rating(fen, moves):
    features = extract_features(fen, moves)
    return model.predict([features])[0]

# Example prediction
example_fen = "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17"
example_moves = "e8d7 a2e6 d7d8 f7f8"
print(f"Predicted Rating: {predict_rating(example_fen, example_moves)}")
