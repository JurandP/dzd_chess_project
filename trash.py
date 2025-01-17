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

# Chess utilities
def material_balance(board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    balance = 0
    for piece in piece_values:
        balance += piece_values[piece] * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
    return balance

def count_open_files(board):
    open_files = 0
    for file in range(8):
        squares = [chess.square(file, rank) for rank in range(8)]
        if all(board.piece_type_at(sq) != chess.PAWN for sq in squares):
            open_files += 1
    return open_files

def count_semi_open_files(board):
    semi_open_files = 0
    for file in range(8):
        squares = [chess.square(file, rank) for rank in range(8)]
        white_pawn = any(board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE) for sq in squares)
        black_pawn = any(board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK) for sq in squares)
        if white_pawn != black_pawn:
            semi_open_files += 1
    return semi_open_files

def count_heavy_pieces(board):
    return (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) +
            len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))

def count_light_pieces(board):
    return (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) +
            len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)))

def count_nearby_pieces(board, color):
    king_square = board.king(color)
    nearby_squares = chess.SquareSet(chess.Board().attacks(king_square))
    return sum(1 for sq in nearby_squares if board.piece_at(sq) and board.piece_at(sq).color == color)

def count_enemy_nearby_pieces(board, color):
    king_square = board.king(color)
    nearby_squares = chess.SquareSet(chess.Board().attacks(king_square))
    return sum(1 for sq in nearby_squares if board.piece_at(sq) and board.piece_at(sq).color != color)

# Function to extract features
def extract_features(fen, moves):
    board = chess.Board(fen)
    evals = []
    handmade_features = {}

    for i, move in enumerate(moves.split()):
        try:
            move = chess.Move.from_uci(move)
            board.push(move)
            if i == 0:  # After the first move, calculate handmade features
                handmade_features['material_balance'] = material_balance(board)
                handmade_features['open_files'] = count_open_files(board)
                handmade_features['semi_open_files'] = count_semi_open_files(board)
                handmade_features['heavy_pieces'] = count_heavy_pieces(board)
                handmade_features['light_pieces'] = count_light_pieces(board)
                handmade_features['own_pieces_near_king'] = count_nearby_pieces(board, board.turn)
                handmade_features['enemy_pieces_near_king'] = count_enemy_nearby_pieces(board, not board.turn)

            if i != 0:
                eval_info = engine.analyse(board, chess.engine.Limit(depth=5))
                evals.append(eval_info['score'].relative.score(mate_score=10000))
        except Exception as e:
            print(board)
            print(f"Error processing move {move}: {e}")
            break

    eval_features = [np.mean(evals) if evals else 0, np.std(evals) if evals else 0]
    all_features = eval_features + list(handmade_features.values())
    return all_features

# Load the data
print('loading_data ...')
data = pd.read_csv('../lichess_db_puzzle.csv', nrows=1000)
print('loaded data :D')

# Initialize chess engine204021.2
engine_path = "../stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Process the dataset
tqdm.pandas()
data['features'] = data.progress_apply(lambda row: extract_features(row['FEN'], row['Moves']), axis=1)
print('features extracted :D')

# Expand features into separate columns
feature_columns = ['mean_eval', 'std_eval', 'material_balance', 'open_files', 'semi_open_files',
                   'heavy_pieces', 'light_pieces', 'own_pieces_near_king', 'enemy_pieces_near_king']
data[feature_columns] = pd.DataFrame(data['features'].tolist(), index=data.index)
data = data.drop(columns=['features'])

# Prepare training and test data
X = data[feature_columns]
y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_test.head())

print('data prepared :D')
print('training model ...')

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print('model trained :D')

# Evaluate the model
predictions = model.predict(X_test_scaled)
print(f'X_test:\n{X_test_scaled[:5]}')
print(f'y_test:\n{y_test.head()}')
print(f'predictions:\n{predictions[:5]}')
mse = mean_squared_error(y_test, predictions)
print(f" Mean Squared Error: {mse:.2f}")
sanity_mse = mean_squared_error(y_test, np.ones_like(y_test) * np.mean(y_train))
print(f" Sanity MSE: {sanity_mse:.2f}")

# Save the model
joblib.dump(model, 'local/linear_chess_rating_model.pkl')
joblib.dump(scaler, 'local/scaler.pkl')

# Prediction example
def predict_rating(fen, moves):
    features = extract_features(fen, moves)
    return model.predict([features])[0]

# Example usage
example_fen = "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17"
example_moves = "e8d7 a2e6 d7d8 f7f8"
print(f"Predicted Rating: {predict_rating(example_fen, example_moves)}")

# Close the engine
engine.quit()
