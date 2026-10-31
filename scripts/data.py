import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing.data_splits import player_id_from_stem
from preprocessing.precompute_annotations import clip_stem, MUTE_CLASS

CQT_DIR = "output/processed_cqt"
LABEL_DIR = "output/per_frame_annotations"
CONTEXT = 9                 # frames of CQT the model sees per prediction
HALF = CONTEXT // 2
N_FRETS = 25

TRAIN_PLAYERS = ["00", "01", "02", "03"]
VAL_PLAYERS = ["04"]
TEST_PLAYERS = ["05"]
ALL_PLAYERS = ["00", "01", "02", "03", "04", "05"]


def tab_to_fretboard(tab):
    """(frames, 6) fret numbers -> (frames, 6, 25) 0/1 grid. Muted strings stay all zeros."""
    """converts 2D matrix of frame number as row index, string number as string, and tab[frame, string]"""
    """being what tab being played on a certain string in a certain frame"""

    out = np.zeros((len(tab), 6, N_FRETS), dtype=np.float32) # creates 3D array of frames * 6 * 25, with datatype of 32-bit float
    frames, strings = np.nonzero(tab != MUTE_CLASS) # tab is a 2D matrix, so this unpacks the 2D array into an array of row indices (frames), 
                                                    # and array of column indices (strings)
    out[frames, strings, tab[frames, strings]] = 1 

    return out


def load_clips(players):
    """Returns a list of (cqt, fretboard, stem) for every recording by these players."""

    clips = []
    for path in sorted(glob.glob(f"{CQT_DIR}/*.npy")): # for every .npy file in output/processed_cqt

        stem = clip_stem(Path(path).stem) # converts path into usable Path object, then .stem returns returns path without extension (i.e w/out .npy)
        if player_id_from_stem(stem) not in players: # check if player is a valid player, if not skip this file. check may not be useful, as data is validated
            continue

        cqt = np.log1p(np.load(path)).astype(np.float32) # loads whatever is in path, calculates the natural logarithm of 1 + x element-wise, and then casts to 32-bit float
        tab = np.load(f"{LABEL_DIR}/{stem}.npz")["Y_tab"] # loads Y_tab of output/per_frame_annotations/stem (define above)
        n = min(len(cqt), len(tab)) # takes minimum of len of cqt container and tab container
        clips.append((cqt[:n], tab_to_fretboard(tab[:n]), Path(path).stem)) # takes the first n entries in cqt, converts first n entries of tab to 3D, binary matrix
                                                                            # either the whole cqt array or the whle tab array is appended.

    return clips


class FrameWindows(tf.keras.utils.Sequence):
    """Batches of (CQT window around frame t, fretboard at frame t)."""

    def __init__(self, clips, batch_size=128, shuffle=True):
        super().__init__()
        self.cqts = [np.pad(c, ((HALF, HALF), (0, 0))) for c, _, _ in clips] 

        """
        c here is a single spectrogram
        ((top, bottom), (left, right))
        Without Padding (Fails at the edge):
        [Window] -> [? ? ? ? | Frame 0, Frame 1, Frame 2, Frame 3...] 
              ^ Error: No data exists before Frame 0!
              for frames in the beginning of the frame, there are no frames before
              and for frames towards the end of the frame, there are no frames after

        With Padding (Succeeds at the edge):
        [Window] -> [0 0 0 0 | Frame 0, Frame 1, Frame 2, Frame 3...]
                       ^ Frame 0 is now safely centered in the window!
        """

        self.labels = [y for _, y, _ in clips]

        self.index = np.concatenate([
            np.stack([np.full(len(y), i), np.arange(len(y))], axis=1)
            for i, y in enumerate(self.labels)
        ])

        """np.full(len(y), i) creates an array of len(y) with all values filed w current i (corresponding to which clip)"""
        """np.arange(len(y)) creates an array of 0, 1, 2, 3, 4... until len(y) (corresponding to frame number)"""
        """result index is a stack of [clip_id, frame_id] for every clip - handles varying clip lengths/number of frames"""

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.index) / self.batch_size))

    def __getitem__(self, b):
        idx = self.index[b * self.batch_size:(b + 1) * self.batch_size] # pulls 128 random frames from random clips. clips are randomized because adjacent clips in audio files
                                                                        # are similar, so model may overfit - ?
                                                                        # If a batch contains 128 consecutive frames from the same song, 
                                                                        # the gradient updates will be highly redundant, causing the model to overfit or learn incredibly slowly.
        x = np.stack([self.cqts[i][t:t + CONTEXT].T for i, t in idx])[..., None]    # (B, 252, 9, 1)
                                                                                    # creates 9 frames centered around i + CONTEXT / 2 for every i in idx
        y = np.stack([self.labels[i][t] for i, t in idx])                           # (B, 6, 25)
                                                                                    # create stack of ground truth labels for each of the 9 frames for every i in idx
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index)
