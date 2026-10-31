import itertools
import numpy as np

TUNING = [40, 45, 50, 55, 59, 64]   # open-string MIDI notes, low E -> high E (row order of p)
N_FRETS = 25                        # frets 0..24
MAX_SPAN = 5                        # fretted notes must fit in a 6-fret window (max - min <= 5)
LOW, HIGH = TUNING[0], TUNING[-1] + N_FRETS - 1


def fold_into_range(pitches):
    """Step 1: shift each pitch by octaves until the guitar can play it."""
    out = []
    for m in pitches:
        while m < LOW:
            m += 12
        while m > HIGH:
            m -= 12
        out.append(m)
    return sorted(set(out))         # folding can create duplicates


def placements(pitches):
    """Step 4: every way to put each pitch on its own string, as {string: fret}."""
    results = []

    def place(i, used):
        if i == len(pitches):
            frets = [f for f in used.values() if f > 0]   # open strings don't count toward the stretch
            if not frets or max(frets) - min(frets) <= MAX_SPAN:
                results.append(dict(used))
            return
        for s, open_note in enumerate(TUNING):
            fret = pitches[i] - open_note
            if s not in used and 0 <= fret < N_FRETS:     # at most one pitch per string
                used[s] = fret
                place(i + 1, used)
                del used[s]

    place(0, {})
    return results


def to_binary(placement):
    b = np.zeros((6, N_FRETS))
    for s, f in placement.items():
        b[s, f] = 1
    return b


def best_fingering(pitches, p):
    """pitches: list of MIDI notes. p: (6, 25) network output. Returns (6, 25) binary tab."""
    pitches = fold_into_range(pitches)
    if not pitches:
        return np.zeros((6, N_FRETS))

    for n in range(min(len(pitches), 6), 0, -1):              # steps 2 and 5
        candidates = [to_binary(pl)
                      for combo in itertools.combinations(pitches, n)   # step 3
                      for pl in placements(list(combo))]
        if candidates:
            scores = [np.sum(p * b) for b in candidates]      # inner product p · b_k
            return candidates[int(np.argmax(scores))]         # eq. (1)

    return np.zeros((6, N_FRETS))
