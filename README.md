# Chess Puzzle Rating Predictor

This project uses a chess engine (Stockfish) and a machine learning model to predict the rating of chess puzzles based on their FEN (Forsyth-Edwards Notation) and solution moves. The linear regression model is trained using features extracted from Stockfish evaluations.

## Requirements

- Python 3.9 or later
- Conda for environment management
- Stockfish chess engine (free and open-source)

## Installation and Setup

### Step 1: Download and Set Up Stockfish

1. Visit the [Stockfish download page](https://stockfishchess.org/download/).
2. Identify the version suitable for your CPU. For example, if your CPU supports AVX2, download the `AVX2` version.
3. Extract the downloaded file:
   ```bash
   tar -xvf stockfish-ubuntu-x86-64-avx2.tar
   ```
4. Note the path to the `stockfish` binary, typically:
   ```
   /path/to/extracted/directory/stockfish
   ```

### Step 2: Create and Activate Conda Environment

1. Create a new Conda environment:
   ```bash
   conda create --name chess_env python=3.9 -y
   ```
2. Activate the environment:
   ```bash
   conda activate chess_env
   ```

### Step 3: Install Dependencies

1. Install the required Python packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Verify Stockfish Path in the Script

Update the `engine_path` variable in the script with the full path to your `stockfish` binary:
```python
engine_path = "/path/to/extracted/directory/stockfish"
```

## Running the Script

1. Ensure all dependencies are installed and the Stockfish binary is accessible.
2. Run the script to train the model and predict puzzle ratings.
3. Example usage for prediction:
   ```python
   example_fen = "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17"
   example_moves = "e8d7 a2e6 d7d8 f7f8"
   print(f"Predicted Rating: {predict_rating(example_fen, example_moves)}")
   ```

## Notes

- Ensure your Stockfish binary matches your CPU capabilities (e.g., AVX2 for modern CPUs).
- You can monitor the feature extraction progress using a `tqdm` progress bar by enabling:
   ```python
   from tqdm import tqdm
   tqdm.pandas()
   data['features'] = data.progress_apply(lambda row: extract_features(row['FEN'], row['Moves']), axis=1)
   
