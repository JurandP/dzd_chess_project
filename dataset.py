import pandas as pd
import chess
import chess.engine
from sklearn.model_selection import train_test_split
import os
import time
import argparse

from dataset_utils import get_features


def main(config):
    start_time = time.time()
    n_rows = config.n_rows
    # Load the data
    print("loading_data ...")
    data = pd.read_csv("../lichess_db_puzzle.csv", nrows=n_rows)
    print("loaded data :D")

    # Initialize chess engine204021.2
    engine_path = (
        "../stockfish-ubuntu-x86-64-avx2/stockfish/stockfish-ubuntu-x86-64-avx2"
    )
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    X, y = get_features(df=data, chess_engine=engine)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Saving dataset...")
    dataset_folder = f"../dataset/{n_rows}"
    os.makedirs(dataset_folder, exist_ok=True)
    X_train.to_csv(f"{dataset_folder}X_train.csv", index=False)
    X_test.to_csv(f"{dataset_folder}X_test.csv", index=False)
    y_train.to_csv(f"{dataset_folder}y_train.csv", index=False)
    y_test.to_csv(f"{dataset_folder}y_test.csv", index=False)
    print("Dataset saved.")

    engine.close()
    time_taken = (time.time() - start_time) / 60
    print(f"creating dataset from {n_rows} rows took {time_taken:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=5000)
    config = parser.parse_args()
    main(config)
