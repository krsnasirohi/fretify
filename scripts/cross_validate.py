"""6-fold cross-validation: each player takes a turn as the test set.

For each fold: test on one player, validate on the next, train on the other four.
Results are saved after every fold, so you can stop with Ctrl+C and rerun later;
finished folds are skipped (an unfinished fold restarts from scratch).

Run from the project root:  python -m scripts.cross_validate
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.data import ALL_PLAYERS
from scripts.test import evaluate, print_results
from scripts.train import train

RESULTS_PATH = Path("output/cv_results.json")
CHECKPOINT_DIR = Path("checkpoints/cv")


def main():
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}

    for k, test_player in enumerate(ALL_PLAYERS):
        if test_player in results:
            print(f"Fold {k + 1}/6 (test player {test_player}): already done, skipping")
            continue

        val_player = ALL_PLAYERS[(k + 1) % len(ALL_PLAYERS)]
        train_players = [p for p in ALL_PLAYERS if p not in (test_player, val_player)]
        print(f"\n=== Fold {k + 1}/6: test {test_player}, val {val_player}, train {train_players}")

        checkpoint = CHECKPOINT_DIR / f"fold_{test_player}.keras"
        train(train_players, [val_player], checkpoint)

        tf.keras.backend.clear_session()                 # free the training model's memory
        model = tf.keras.models.load_model(checkpoint)   # best epoch, not the last one
        results[test_player] = evaluate(model, [test_player])
        print_results(results[test_player])

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== Per-player F1")
    methods = list(next(iter(results.values())).keys())
    for player in ALL_PLAYERS:
        print(f"  player {player}: " + "  ".join(f"{m}: {results[player][m][2]:.3f}" for m in methods))

    print("\n=== Average over 6 folds (mean ± std)")
    for m in methods:
        scores = np.array([results[player][m] for player in ALL_PLAYERS])   # (6, 3)
        mean, std = scores.mean(axis=0), scores.std(axis=0)
        print(f"{m:30s} precision {mean[0]:.3f}±{std[0]:.3f}  "
              f"recall {mean[1]:.3f}±{std[1]:.3f}  F1 {mean[2]:.3f}±{std[2]:.3f}")


if __name__ == "__main__":
    main()
