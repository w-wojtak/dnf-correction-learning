# pylint: disable=C0200

from pathlib import Path
from datetime import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # type: ignore

# requires ffmpeg installed on your system
from matplotlib.animation import FFMpegWriter

from utils import *


# skip inputs
# 6 actions
# EARLY
# LATE
# BEFORE
# AFTER
# SKIP/REMOVE
# ADD


# ====================================
# -------- Project paths -------------
# ====================================

# Project root = current working directory
PROJECT_ROOT = Path.cwd()

# Results directory
results_dir = PROJECT_ROOT / "results_tuning"
results_dir.mkdir(exist_ok=True)


# ====================================
# -------- Load memory ---------------
# ====================================

# folder = "/home/wwojtak/dnf_architecture_python/data_basic"
folder = PROJECT_ROOT / "results_learning"

file1, ts1 = find_latest_file_with_prefix(folder, "u_field_1_")
# file2, ts2 = find_latest_file_with_prefix(folder, "u_field_2_")
file3, ts3 = find_latest_file_with_prefix(folder, "u_d_")

# Optional: check if all timestamps match
if ts1 != ts3:
    raise ValueError("Timestamps do not match across all files.")

# Load data
u_field_1 = np.load(file1)
# u_field_2 = np.load(file2)
u_d = np.load(file3)


# ====================================
# -------- Initialization ------------
# ====================================

x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05
# theta = 1

# x = np.linspace(-x_lim, x_lim, 200)
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)


# Positions for input set 1
input_position_1 = [-60, -30, 0, 30, 60]

input_positions = input_position_1
input_indices = [np.argmin(np.abs(x - pos)) for pos in input_positions]





# ====================================
# --- Load fields for initialization --
# ====================================

try:
    # Flatten and compute initial h from task duration
    u_d = u_d.flatten()
    h_d_initial = max(u_d)

    # Use u_field_1 for u_act
    # u_act = u_field_1.flatten() - h_d_initial + 1.5
    u_act = u_field_1 - h_d_initial + 1.5
    input_action_onset = u_field_1.flatten()
    h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

    # Use u_field_2 for u_sim
    u_sim = u_field_1.flatten() - h_d_initial + 1.5
    input_action_onset_2 = u_field_1.flatten()
    h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

except FileNotFoundError:
    print("No previous sequence memory found, initializing with default values.")
    u_act = np.zeros(np.shape(x))
    h_u_act = -3.2 * np.ones(np.shape(x))


# ====================================
# -------- Field parameters ----------
# ====================================

# needed: u act and u wm

kernel_pars_act = [1.5, 0.8, 0.1]   # ok ADDED INHIBITION 0.5
kernel_pars_wm = [1.75, 0.5, 0.8]   # ok

# Working memory parameters
h_0_wm = -1.0
theta_wm = 0.8

u_wm = h_0_wm * np.ones(np.shape(x))
h_u_wm = h_0_wm * np.ones(np.shape(x))

# Action onset parameters
tau_h_act = 20
theta_act = 1.5


# ====================================
# -------- Kernels & FFTs ------------
# ====================================

kernel_act = kernel_gauss(x, *kernel_pars_act)
kernel_wm = kernel_osc(x, *kernel_pars_wm)

w_hat_act = np.fft.fft(kernel_act)
w_hat_wm = np.fft.fft(kernel_wm)


# ====================================
# -------- History storage -----------
# ====================================

u_act_history = []   # Lists to store values at each time step
u_wm_history = []

# Adaptation memory field
h_u_amem = np.zeros(np.shape(x))


# ====================================
# -------- Online plotting -----------
# ====================================

plot_fields = True
plot_every = 5      # update plot every x time steps
plot_delay = 0.05   # delay (in seconds) before each plot update

fig = axs = None
line_u_act = line_u_wm = None

if plot_fields:
    plt.ion()

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top: u_act ---
    line_u_act, = axs[0].plot(x, u_act, label="u_act(x)")
    axs[0].set_ylim(-5, 5)
    axs[0].set_ylabel("Activity")
    axs[0].legend()
    axs[0].set_title("Field u_act - Time = 0")

    # --- Bottom: u_wm ---
    line_u_wm, = axs[1].plot(x, u_wm, label="u_wm(x)")
    axs[1].set_ylim(-5, 5)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Activity")
    axs[1].legend()
    axs[1].set_title("Field u_wm - Time = 0")

    plt.tight_layout()


# ====================================
# -------- Recall simulation ---------
# ====================================

# MAIN LOOP
for i in range(len(t)):

    f_act = np.heaviside(u_act - theta_act, 1)
    f_hat_act = np.fft.fft(f_act)
    conv_act = (
        dx
        * np.fft.ifftshift(
            np.real(np.fft.ifft(f_hat_act * w_hat_act))
        )
    )

    f_wm = np.heaviside(u_wm - theta_wm, 1)
    f_hat_wm = np.fft.fft(f_wm)
    conv_wm = (
        dx
        * np.fft.ifftshift(
            np.real(np.fft.ifft(f_hat_wm * w_hat_wm))
        )
    )

    # Update field states
    h_u_act += dt / tau_h_act

    u_act += dt * (
        -u_act
        + conv_act
        + input_action_onset
        + h_u_act
        - 6.0 * f_wm * conv_wm
    )

    u_wm += (dt / 1.25) * (
        -u_wm
        + conv_wm
        + 8 * (f_act * u_act)
        + h_u_wm
    )

    # Store the values at the specified positions in history arrays
    u_act_values_at_positions = [u_act[idx] for idx in input_indices]
    u_act_history.append(u_act_values_at_positions)

    u_wm_values_at_positions = [u_wm[idx] for idx in input_indices]
    u_wm_history.append(u_wm_values_at_positions)

        # ------------------------------------
    # Online plot update
    # ------------------------------------
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):

        line_u_act.set_ydata(u_act)
        line_u_wm.set_ydata(u_wm)

        axs[0].set_title(f"Field u_act - Time = {i}")
        axs[1].set_title(f"Field u_wm - Time = {i}")

        plt.pause(plot_delay)

