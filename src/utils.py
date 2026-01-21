import os
import numpy as np
from datetime import datetime
from pathlib import Path


def kernel_osc(x, a, b, alpha):
    return a * (np.exp(-b * abs(x)) * (b * np.sin(abs(alpha * x)) + np.cos(alpha * x)))


def kernel_gauss(x, a_ex, s_ex, w_in):
    """
    Computes a Gaussian kernel minus inhibition.

    Parameters:
    - x (np.ndarray): Spatial positions.
    - a_ex (float): Amplitude of excitation.
    - s_ex (float): Width (standard deviation) of excitation.
    - w_in (float): Global inhibition strength.

    Returns:
    - np.ndarray: Kernel values.
    """
    return a_ex * np.exp(-0.5 * x**2 / s_ex**2) - w_in


def get_inputs(x, t, dt, input_pars, input_flag):
    if not input_flag:
        return np.zeros((len(t), len(x)))

    [input_shape, input_positions, input_onsets, input_durations] = input_pars
    amplitude, sigma = input_shape
    inputs = np.zeros((len(t), len(x)))

    for pos, onset, dur in zip(input_positions, input_onsets, input_durations):
        time_on = int(onset / dt)
        time_off = int((onset + dur) / dt)
        gaussian = amplitude * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
        inputs[time_on:time_off, :] += gaussian

    return inputs


def find_latest_file_with_prefix(folder_path, prefix):
    folder_path = Path(folder_path)
    files = os.listdir(folder_path)

    matching_files = sorted([
        f for f in files if f.startswith(prefix) and f.endswith(".npy")
    ])

    if not matching_files:
        raise FileNotFoundError(f"No files found with prefix {prefix}")

    # Extract timestamps: assumes format like 'u_field_1_YYYYMMDD_HHMMSS.npy'
    def extract_timestamp(fname):
        return "_".join(fname.split("_")[-2:]).replace(".npy", "")

    timestamps = [extract_timestamp(f) for f in matching_files]
    latest_ts = sorted(timestamps)[-1]
    latest_filename = f"{prefix}{latest_ts}.npy"

    return folder_path / latest_filename, latest_ts