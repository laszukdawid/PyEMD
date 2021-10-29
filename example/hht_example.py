import numpy as np
import pylab as plt
from scipy.signal import hilbert

from PyEMD import EMD


def instant_phase(imfs):
    """Extract analytical signal through Hilbert Transform."""
    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row
    phase = np.unwrap(np.angle(analytic_signal))  # Compute angle between img and real
    return phase


# Define signal
t = np.linspace(0, 1, 200)
dt = t[1] - t[0]

sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
S = 3 * sin(18, 0.2) * (t - 0.2) ** 2
S += 5 * sin(11, 2.7)
S += 3 * sin(14, 1.6)
S += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
S += t ** 2.1 - t

# Compute IMFs with EMD
emd = EMD()
imfs = emd(S, t)

# Extract instantaneous phases and frequencies using Hilbert transform
instant_phases = instant_phase(imfs)
instant_freqs = np.diff(instant_phases) / (2 * np.pi * dt)

# Create a figure consisting of 3 panels which from the top are the input signal, IMFs and instantaneous frequencies
fig, axes = plt.subplots(3, figsize=(12, 12))

# The top panel shows the input signal
ax = axes[0]
ax.plot(t, S)
ax.set_ylabel("Amplitude [arb. u.]")
ax.set_title("Input signal")

# The middle panel shows all IMFs
ax = axes[1]
for num, imf in enumerate(imfs):
    ax.plot(t, imf, label="IMF %s" % (num + 1))

# Label the figure
ax.legend()
ax.set_ylabel("Amplitude [arb. u.]")
ax.set_title("IMFs")

# The bottom panel shows all instantaneous frequencies
ax = axes[2]
for num, instant_freq in enumerate(instant_freqs):
    ax.plot(t[:-1], instant_freq, label="IMF %s" % (num + 1))

# Label the figure
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Inst. Freq. [Hz]")
ax.set_title("Huang-Hilbert Transform")

plt.tight_layout()
plt.savefig("hht_example", dpi=120)
plt.show()
