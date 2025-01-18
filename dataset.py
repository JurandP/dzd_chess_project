import pandas as pd
import chess
import chess.engine
from sklearn.model_selection import train_test_split

from dataset_utils import get_features


# Load the data
print("loading_data ...")
data = pd.read_csv("../lichess_db_puzzle.csv", nrows=5000)
print("loaded data :D")

# Initialize chess engine204021.2
engine_path = "../stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

X, y = get_features(df=data, chess_engine=engine)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Saving dataset...")
dataset_folder = "../dataset/"
X_train.to_csv(f"{dataset_folder}X_train.csv", index=False)
X_test.to_csv(f"{dataset_folder}X_test.csv", index=False)
y_train.to_csv(f"{dataset_folder}y_train.csv", index=False)
y_test.to_csv(f"{dataset_folder}y_test.csv", index=False)
print("Dataset saved.")

engine.close()