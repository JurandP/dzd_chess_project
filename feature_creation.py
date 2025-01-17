from functools import partial
import pandas as pd
import chess
import chess.engine
import numpy as np
from tqdm.auto import tqdm


# Chess utilities
def material_balance(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    balance = 0
    for piece in piece_values:
        balance += piece_values[piece] * (
            len(board.pieces(piece, chess.WHITE))
            - len(board.pieces(piece, chess.BLACK))
        )
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
        white_pawn = any(
            board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE) for sq in squares
        )
        black_pawn = any(
            board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK) for sq in squares
        )
        if white_pawn != black_pawn:
            semi_open_files += 1
    return semi_open_files


def count_heavy_pieces(board):
    return (
        len(board.pieces(chess.ROOK, chess.WHITE))
        + len(board.pieces(chess.ROOK, chess.BLACK))
        + len(board.pieces(chess.QUEEN, chess.WHITE))
        + len(board.pieces(chess.QUEEN, chess.BLACK))
    )


def count_light_pieces(board):
    return (
        len(board.pieces(chess.KNIGHT, chess.WHITE))
        + len(board.pieces(chess.KNIGHT, chess.BLACK))
        + len(board.pieces(chess.BISHOP, chess.WHITE))
        + len(board.pieces(chess.BISHOP, chess.BLACK))
    )


def count_nearby_pieces(board, color):
    king_square = board.king(color)
    nearby_squares = chess.SquareSet(chess.Board().attacks(king_square))
    return sum(
        1
        for sq in nearby_squares
        if board.piece_at(sq) and board.piece_at(sq).color == color
    )


def count_enemy_nearby_pieces(board, color):
    king_square = board.king(color)
    nearby_squares = chess.SquareSet(chess.Board().attacks(king_square))
    return sum(
        1
        for sq in nearby_squares
        if board.piece_at(sq) and board.piece_at(sq).color != color
    )


# Function to extract features

def extract_features(fen, moves, chess_engine):
    board = chess.Board(fen)
    evals_depth_5 = []
    evals_depth_10 = []
    deltas = []
    handmade_features = {}
    mate_threats_depth_5 = []
    mate_threats_depth_10 = []

    for i, move in enumerate(moves.split()):
        try:
            move = chess.Move.from_uci(move)
            board.push(move)

            # Analysis at depth 5
            eval_info_depth_5 = chess_engine.analyse(board, chess.engine.Limit(depth=5))
            eval_depth_5 = eval_info_depth_5['score'].relative.score(mate_score=10000)
            evals_depth_5.append(eval_depth_5)

            # Analysis at depth 10
            eval_info_depth_10 = chess_engine.analyse(board, chess.engine.Limit(depth=10))
            eval_depth_10 = eval_info_depth_10['score'].relative.score(mate_score=10000)
            evals_depth_10.append(eval_depth_10)

            # Evaluate deltas (change in evaluation between moves at depth 5)
            if len(evals_depth_5) > 1:
                deltas.append(abs(evals_depth_5[-1] - evals_depth_5[-2]))

            # Mate in X detection
            if eval_info_depth_5['score'].relative.is_mate():
                mate_threats_depth_5.append(eval_info_depth_5['score'].relative.mate())
            if eval_info_depth_10['score'].relative.is_mate():
                mate_threats_depth_10.append(eval_info_depth_10['score'].relative.mate())

            # Handmade features after first move
            if i == 0:
                handmade_features['material_balance'] = material_balance(board)
                handmade_features['open_files'] = count_open_files(board)
                handmade_features['semi_open_files'] = count_semi_open_files(board)
                handmade_features['heavy_pieces'] = count_heavy_pieces(board)
                handmade_features['light_pieces'] = count_light_pieces(board)
                handmade_features['own_pieces_near_king'] = count_nearby_pieces(board, board.turn)
                handmade_features['enemy_pieces_near_king'] = count_enemy_nearby_pieces(board, not board.turn)
        except Exception as e:
            print(f"Error processing move {move}: {e}")
            break

    # Evaluation features for depth 5
    eval_features_depth_5 = [
        np.mean(evals_depth_5) if evals_depth_5 else 0,
        np.std(evals_depth_5) if evals_depth_5 else 0,
        evals_depth_5[0] if evals_depth_5 else 0,  # First evaluation
        evals_depth_5[-1] if evals_depth_5 else 0,  # Last evaluation
        max(evals_depth_5) if evals_depth_5 else 0,  # Maximum evaluation
        min(evals_depth_5) if evals_depth_5 else 0,  # Minimum evaluation
        evals_depth_5[-1] - evals_depth_5[0] if len(evals_depth_5) > 1 else 0,  # Evaluation difference
        np.mean(deltas) if deltas else 0,  # Average delta
        len([d for d in deltas if d > 300])  # Blunder count
    ]

    # Evaluation features for depth 10
    eval_features_depth_10 = [
        np.mean(evals_depth_10) if evals_depth_10 else 0,
        np.std(evals_depth_10) if evals_depth_10 else 0,
        evals_depth_10[0] if evals_depth_10 else 0,  # First evaluation
        evals_depth_10[-1] if evals_depth_10 else 0,  # Last evaluation
        max(evals_depth_10) if evals_depth_10 else 0,  # Maximum evaluation
        min(evals_depth_10) if evals_depth_10 else 0,  # Minimum evaluation
        evals_depth_10[-1] - evals_depth_10[0] if len(evals_depth_10) > 1 else 0  # Evaluation difference
    ]

    # Mate threat features
    mate_features_depth_5 = [
        mate_threats_depth_5[0] if mate_threats_depth_5 else 0,  # First mate threat at depth 5
        mate_threats_depth_5[-1] if mate_threats_depth_5 else 0  # Last mate threat at depth 5
    ]

    mate_features_depth_10 = [
        mate_threats_depth_10[0] if mate_threats_depth_10 else 0,  # First mate threat at depth 10
        mate_threats_depth_10[-1] if mate_threats_depth_10 else 0  # Last mate threat at depth 10
    ]

    # Combine features
    all_features = (
        eval_features_depth_5 +
        mate_features_depth_5 +
        eval_features_depth_10 +
        mate_features_depth_10 +
        list(handmade_features.values())
    )
    return all_features


def get_features(df, chess_engine):
    extract = partial(extract_features, chess_engine=chess_engine)
    # Process the dataset to add features
    tqdm.pandas()
    df['features'] = df.progress_apply(lambda row: extract(row['FEN'], row['Moves']), axis=1)
    print('features extracted :D')

    feature_columns = [
        "mean_eval",
        "std_eval",
        "material_balance",
        "open_files",
        "semi_open_files",
        "heavy_pieces",
        "light_pieces",
        "own_pieces_near_king",
        "enemy_pieces_near_king",
    ]
    df[feature_columns] = pd.DataFrame(df["features"].tolist(), index=df.index)
    df = df.drop(columns=["features"])

    # Prepare training and test df
    X = df[feature_columns]
    y = df['Rating']

    return X, y