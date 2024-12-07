import numpy as np

# Spectrogram dimensions
height, width = 79, 534

# Function to generate a power spectrogram with synthetic patterns and noise
def generate_power_spectrogram(height, width):
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    # Initialize spectrogram with background noise around -105 dB
    spectrogram = -105 + 10 * np.random.randn(height, width)

    # Add spots of even lower values (around -140 dB)
    low_spots = np.random.choice([0, 1], size=(height, width), p=[0.98, 0.02])
    spectrogram[low_spots == 1] = -140

    # Add horizontal line with values around -60 dB, adding noise of ±5 dB
    horizontal_mask = np.abs(Y - 0.7) < 0.02
    spectrogram[horizontal_mask] = -60 + 5 * np.random.randn(np.sum(horizontal_mask))
    spectrogram[horizontal_mask] += np.random.uniform(-5, 5, np.sum(horizontal_mask))

    # Add diagonal line with values around -60 dB, adding noise of ±5 dB (higher slope, starting at ~200 s, and ending at horizontal intersection)
    diagonal_mask = (X >= 200 / 534) & (Y >= 2 * X - 0.4) & (Y <= 0.7) & (np.abs(Y - (2 * X - 0.4)) < 0.02)  # Slope = 2, y-intercept = -0.4
    spectrogram[diagonal_mask] = -60 + 5 * np.random.randn(np.sum(diagonal_mask))
    spectrogram[diagonal_mask] += np.random.uniform(-5, 5, np.sum(diagonal_mask))

    return spectrogram

# Function to generate a phase spectrogram with synthetic patterns and noise
def generate_phase_spectrogram(height, width):
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    # Initialize spectrogram with background values around 0
    spectrogram = np.zeros((height, width))

    # Add horizontal line with random values around -pi to pi
    horizontal_mask = np.abs(Y - 0.7) < 0.02
    spectrogram[horizontal_mask] = np.random.uniform(-np.pi, np.pi, np.sum(horizontal_mask))

    # Add diagonal line with random values around -pi to pi (higher slope, starting at ~200 s, and ending at horizontal intersection)
    diagonal_mask = (X >= 200 / 534) & (Y >= 2 * X - 0.4) & (Y <= 0.7) & (np.abs(Y - (2 * X - 0.4)) < 0.02)  # Slope = 2, y-intercept = -0.4
    spectrogram[diagonal_mask] = np.random.uniform(-np.pi, np.pi, np.sum(diagonal_mask))

    return spectrogram

# Generate 4 power and 2 phase spectrograms
power_spectrograms = [generate_power_spectrogram(height, width) for _ in range(4)]
phase_spectrograms = [generate_phase_spectrogram(height, width) for _ in range(2)]

# Stack all spectrograms together (channels 1-4 power, channels 5-6 phase)
stacked_spectrogram = np.stack(power_spectrograms + phase_spectrograms, axis=-1)

# Save the stacked spectrogram as a .npy file
np.save('stacked_spectrogram_train_obs_3.npy', stacked_spectrogram)
