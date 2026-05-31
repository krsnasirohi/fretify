import numpy as np
from scipy.io.wavfile import write

# Parameters
fs = 22050       # Sampling frequency
duration = 1.0   # 1 second
freq = 440.0     # Frequency for A4

# Generate sine wave
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
amplitude = np.iinfo(np.int16).max
data = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)

# Save as 'test_note.wav'
write('test_note.wav', fs, data)
print("Test file 'test_note.wav' created at 440Hz (A4)")