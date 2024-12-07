from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
from typing import Union

from custom_transforms import _to_tensor

def plot_simple_spectrogram(spectrogram: Union[Tensor, str, Path, np.ndarray],
                            spectrogram_channel: int = 0) -> None:
    """Plot a simple spectrogram without annotations.

    Args:
        spectrogram (Tensor | str | Path | np.ndarray): Input spectrogram to plot.
        spectrogram_channel (int, optional): Which of the six spectrogram channels
            to plot. Defaults to 0.
    """
    TS_CROPTWIDTH = (-150, 200)
    VR_CROPTWIDTH = (-60, 15)

    # Load and preprocess the spectrogram
    if isinstance(spectrogram, str):
        spectrogram = Path(spectrogram)

    if isinstance(spectrogram, Path):
        spectrogram = np.load(spectrogram)

    if isinstance(spectrogram, np.ndarray):
        spectrogram = _to_tensor(spectrogram)

    if isinstance(spectrogram, Tensor):
        spectrogram = spectrogram.squeeze()

    # Extract the selected channel
    spectrogram = spectrogram[spectrogram_channel, :, :]

    # Determine color scale based on channel
    if spectrogram_channel < 4:
        if spectrogram_channel < 0:
            raise IndexError("Channel number must be between 0 and 5")
        vmin, vmax = -110, -40  # Adjust based on expected data range
    else:
        if spectrogram_channel > 5:
            raise IndexError("Channel number must be between 0 and 5")
        vmin, vmax = -np.pi, np.pi  # Adjust for angular data

    # Plotting
    _, ax = plt.subplots(1, 1, figsize=(12, 6))
    img = ax.imshow(spectrogram, aspect='auto',
                    extent=[TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000,
                            VR_CROPTWIDTH[0], VR_CROPTWIDTH[1]],
                    vmin=vmin, vmax=vmax,
                    origin='lower',
                    interpolation='bilinear',
                    cmap='jet')  # Choose colormap as needed

    # Add color bar for intensity scale
    cbar = plt.colorbar(img, ax=ax, orientation='vertical')
    cbar.set_label('Intensity (dB)' if spectrogram_channel < 4 else 'Phase (radians)')

    # Set axis labels and grid
    ax.set_ylabel('Radial Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Set title for clarity
    ax.set_title(f'Spectrogram (Channel {spectrogram_channel + 1})')

    plt.tight_layout()
    plt.show()



def plot_spectrogram_with_annotations(spectrogram: Union[Tensor, str, Path, np.ndarray],
                                      target_vr: float,
                                      estimated_vr: float,
                                      spectrogram_channel: int = 0) -> None:
    """Plot spectrogram with true and predicted vr annotation

    Args:
        spectrogram (Tensor | str | Path | np.ndarray): Input spectrogram to plot.
        target_vr (float): True radial ball velocity.
        estimated_vr (float): Predicted radial ball velocity.
        spectrogram_channel (int, optional): Which of the six spectrogram channels
            to plot. Defaults to 0.
    """
    TS_CROPTWIDTH = (-150, 200)
    VR_CROPTWIDTH = (-60, 15)

    # Load and preprocess the spectrogram
    if isinstance(spectrogram, str):
        spectrogram = Path(spectrogram)

    if isinstance(spectrogram, Path):
        spectrogram = np.load(spectrogram)

    if isinstance(spectrogram, np.ndarray):
        spectrogram = _to_tensor(spectrogram)

    if isinstance(spectrogram, Tensor):
        spectrogram = spectrogram.squeeze()

    # Extract the selected channel
    spectrogram = spectrogram[spectrogram_channel, :, :]

    # Determine color scale based on channel
    if spectrogram_channel < 4:
        if spectrogram_channel < 0:
            raise IndexError("Channel number must be between 0 and 5")
        vmin, vmax = -110, -40  # Adjust based on expected data range
    else:
        if spectrogram_channel > 5:
            raise IndexError("Channel number must be between 0 and 5")
        vmin, vmax = -np.pi, np.pi  # Adjust for angular data

    # Plotting
    _, ax = plt.subplots(1, 1, figsize=(12, 6))
    img = ax.imshow(spectrogram, aspect='auto',
                    extent=[TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000,
                            VR_CROPTWIDTH[0], VR_CROPTWIDTH[1]],
                    vmin=vmin, vmax=vmax,
                    origin='lower',
                    interpolation='bilinear',
                    cmap='jet')  # Choose colormap as needed

    # Add color bar for intensity scale
    cbar = plt.colorbar(img, ax=ax, orientation='vertical')
    cbar.set_label('Intensity (dB)' if spectrogram_channel < 4 else 'Phase (radians)')

    # Set axis labels and grid
    ax.set_ylabel('Radial Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Plot annotations for target and estimated radial velocities
    ax.plot([TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000], [target_vr, target_vr], 'r--', linewidth=2, label=r"True $v_{r}$")
    ax.plot([TS_CROPTWIDTH[0] / 1000, TS_CROPTWIDTH[1] / 1000], [estimated_vr, estimated_vr], 'r:', linewidth=2, label=r"Pred. $\bar{v}_{r}$")
    ax.legend()

    # Set title for clarity
    ax.set_title(f'Spectrogram (Channel {spectrogram_channel + 1}) showing target and estimated ' + r'$v_{r}$')

    plt.tight_layout()
    plt.show()
