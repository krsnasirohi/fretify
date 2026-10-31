from pathlib import Path

import tensorflow as tf

from scripts.data import FrameWindows, load_clips, TRAIN_PLAYERS, VAL_PLAYERS
from scripts.pf_cnn import build_model


def train(train_players, val_players, checkpoint, epochs=30):
    """Train a fresh model and save its best epoch (lowest val_loss) to `checkpoint`."""
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    train_data = FrameWindows(load_clips(train_players), batch_size=128, shuffle=True)
    val_data = FrameWindows(load_clips(val_players), batch_size=512, shuffle=False)
    print(f"{len(train_data.index)} train frames, {len(val_data.index)} val frames")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",          # each of the 150 cells is its own yes/no guess
        metrics=[tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.summary()

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[
            # Keep only the version that does best on the validation player
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
            # Stop when it hasn't improved for 10 epochs
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ],
    )
    print(f"Best model saved to {checkpoint}")


if __name__ == "__main__":
    train(TRAIN_PLAYERS, VAL_PLAYERS, "checkpoints/pf_cnn.keras")
