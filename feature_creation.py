from functools import partial
import pandas as pd
import chess
import chess.engine
import numpy as np
from tqdm.auto import tqdm

def extract_features(fen, moves, chess_engine):
    board = chess.Board(fen)
    evals = []
    for i, move in enumerate(moves.split()):
        try:
            move = chess.Move.from_uci(move)
            board.push(move)
            if i != 0:
                eval_info = chess_engine.analyse(board, chess.engine.Limit(depth=5))
                evals.append(eval_info['score'].relative.score(mate_score=10000))
        except Exception as e:
            print(board)
            print(f"Error processing move {move}: {e}")
            break
    # Use mean and std of evaluations as features
    if evals:
        return [np.mean(evals), np.std(evals)]
    return [0, 0]

def get_chess_engine_features(df, chess_engine):
    extract = partial(extract_features, chess_engine=chess_engine)
    # Process the dataset to add features
    tqdm.pandas()
    df['features'] = df.progress_apply(lambda row: extract(row['FEN'], row['Moves']), axis=1)
    print('features extracted :D')
    df[['mean_eval', 'std_eval']] = pd.DataFrame(df['features'].tolist(), index=df.index)
    df = df.drop(columns=['features'])

    # Prepare training and test df
    X = df[['mean_eval', 'std_eval']]
    y = df['Rating']

    return X, y