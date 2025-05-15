import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

# Use actual sample record
record_path = "ptb-xl/records100/01000/01000_lr"

# Load the ECG record
record = wfdb.rdrecord(record_path)
signal = record.p_signal[:, 1][:1000]  # Lead 1, 1000 samples
fs = 100  # Sampling frequency for _lr records

# Plot ECG waveform
plt.figure(figsize=(12, 4))
plt.plot(signal, color='black')
plt.title("Sample ECG Signal - Record 01000_lr (Lead 1)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.tight_layout()
plt.show()

# STFT Spectrogram
f, t, Zxx = stft(signal, fs=fs, nperseg=128, noverlap=64)
magnitude = np.abs(Zxx)

plt.figure(figsize=(10, 4))
plt.pcolormesh(t, f, magnitude, shading='gouraud')
plt.title("STFT Spectrogram of Record 01000_lr (Lead 1)")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s]")
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()
