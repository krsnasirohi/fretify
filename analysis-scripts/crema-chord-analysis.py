import setuptools
import pkg_resources
import crema
from crema.analyze import analyze

# 1. Run the analysis (returns a JAMS object)
# You can pass a path or a librosa-style (y, sr) buffer
jam = analyze(filename='voice.wav')

# 2. Extract the chord annotations
# CREMA provides a list of annotations; index 0 is typically the chord tracker
chords = jam.search(namespace='chord')[0]

# 3. Print detected chords with timestamps
for observation in chords.data:
    start = observation.time
    duration = observation.duration
    label = observation.value
    print(f"[{start:5.2f}s - {start+duration:5.2f}s]: {label}")