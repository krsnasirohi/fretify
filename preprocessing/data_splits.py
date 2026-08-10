"""Shared train/val split helper for GuitarSet-based training scripts.

GuitarSet filenames encode the player as a leading two-digit prefix, e.g.
"00_BN1-129-Eb_comp" -> player "00", shared across all 4 audio variants
(_mix, _mic, _hex, _hex_cln) of a recording and its one .jams annotation.
A plain 80/20 positional split on a sorted file list leaks in two ways: it
can put different mic variants of the same performance on both sides, and
since filenames sort by player, it splits within a player instead of across
players. grouped_split() below fixes both by keeping every file from a given
player on exactly one side of the split.
"""
from typing import Callable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def player_id_from_stem(stem: str) -> str:
    """Extracts the GuitarSet player ID from a filename stem, e.g.
    "00_BN1-129-Eb_comp_hex" -> "00". Player IDs are the two-digit prefix
    before the first underscore, shared by a recording's .jams annotation
    and all 4 of its audio/feature variants.
    """
    return stem.split("_", 1)[0]


def grouped_split(
    items: Sequence[T],
    key_fn: Callable[[T], str],
    val_players: Sequence[str] = None,
    seed: int = 0,
) -> Tuple[List[T], List[T]]:
    """Splits items into (train, val) such that all items sharing a
    key_fn(item) (player ID) land on the same side. Default: hold out the
    player with the largest sorted ID. Pass val_players explicitly to choose
    different holdout player(s), e.g. for cross-validation.
    """
    del seed  # split is deterministic by player ID, not randomized

    players = sorted({key_fn(item) for item in items})
    if val_players is None:
        val_players = players[-1:]
    val_players = set(val_players)

    train_items = [item for item in items if key_fn(item) not in val_players]
    val_items = [item for item in items if key_fn(item) in val_players]
    return train_items, val_items


def print_and_verify_split(
    train_items: Sequence[T],
    val_items: Sequence[T],
    key_fn: Callable[[T], str],
    jams_fn: Callable[[T], str] = None,
) -> None:
    """Prints split diagnostics and asserts the grouping is leak-free.
    jams_fn, if given, extracts a .jams path from an item so we can also
    assert no annotation file appears on both sides.
    """
    n_train, n_val = len(train_items), len(val_items)
    train_players = sorted({key_fn(item) for item in train_items})
    val_players = sorted({key_fn(item) for item in val_items})

    ratio = n_val / (n_train + n_val) if (n_train + n_val) else 0.0
    print(f"Split: {n_train} train / {n_val} val ({ratio:.1%} val)")
    print(f"Train players: {train_players}")
    print(f"Val players:   {val_players}")

    assert set(train_players).isdisjoint(val_players), (
        f"Player leak between train/val: {set(train_players) & set(val_players)}"
    )

    if jams_fn is not None:
        train_jams = {jams_fn(item) for item in train_items}
        val_jams = {jams_fn(item) for item in val_items}
        overlap = train_jams & val_jams
        assert not overlap, f"JAMS file(s) present in both train and val: {overlap}"
