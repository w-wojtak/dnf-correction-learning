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


input_positions2 = np.array(input_positions)

# Midpoints between action centers
boundaries = (input_positions2[:-1] + input_positions2[1:]) / 2

# Add outer boundaries to cover full field
boundaries = np.concatenate((
    [x[0]],
    boundaries,
    [x[-1]]
))


action_buckets = []

for i in range(len(input_positions)):
    left = boundaries[i]
    right = boundaries[i + 1]

    idx = np.where((x >= left) & (x < right))[0]
    action_buckets.append(idx)


action_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
]



# ====================================
# -------- Human feedback ------------
# ====================================

from enum import Enum
from dataclasses import dataclass


class FeedbackType(Enum):
    EARLY = "early"
    LATE = "late"
    BEFORE = "before"
    AFTER = "after"
    SKIP = "skip"
    ADD = "add"


@dataclass
class Annotation:
    time_index: int
    feedback: FeedbackType


class AnnotationBuffer:
    """
    Passive buffer: stores human feedback during execution.
    Does NOT affect dynamics.
    """
    def __init__(self):
        self.annotations = []

    def add(self, annotation: Annotation):
        self.annotations.append(annotation)

    def get_all(self):
        return self.annotations


# ====================================
# -------- Memory editor -------------
# ====================================



class MemoryEditor:
    """
    Applies human feedback AFTER the run
    to modify initial u_act memory.
    """

    def __init__(self, x, action_centers):
        self.x = x
        self.action_centers = np.array(action_centers)

        # Precompute spatial buckets ONCE
        self.action_bounds = self._compute_action_bounds(
            self.x,
            self.action_centers
        )

    def apply_feedback(self, u_act_initial, u_act_history, annotations):
        u_new = u_act_initial.copy()

        for ann in annotations:
            if ann.feedback == FeedbackType.LATE:
                u_new = self._make_later(u_new, ann, u_act_history)

            elif ann.feedback == FeedbackType.EARLY:
                u_new = self._make_earlier(u_new)

            elif ann.feedback == FeedbackType.SKIP:
                u_new = self._remove_peak(
                    u_new,
                    ann.time_index,
                    u_act_history
                )

        return u_new

    def _make_earlier(self, u):
        # shift memory slightly left REMOVE SHIFTING
        #  to make earlier: increase the peak
        #  action's mask (TODO) x f(u_act) -> to increase only the right peak
        #  BUT, how to know which action to edit????? - in annotation we have time step
        # somehow map time intervals to space intervals???
        # and how to decide which one to edit based on when user speaks. the current or past?

        return np.roll(u, -20)
    
    def _action_index_from_time(self, time_index, u_act_history):
        """
        Return which action bucket was active at this time.
        """
        values = u_act_history[time_index]  # shape: (num_actions,)
        return int(np.argmax(values))
    
    def _compute_action_bounds(self, x, centers):
        """
        Compute spatial index ranges for each action bucket.
        """
        bounds = []
        midpoints = (centers[:-1] + centers[1:]) / 2

        left_edges = np.concatenate(([x[0]], midpoints))
        right_edges = np.concatenate((midpoints, [x[-1]]))

        for l, r in zip(left_edges, right_edges):
            idx = np.where((x >= l) & (x < r))[0]
            bounds.append(idx)

        return bounds
    
    def _make_later_action(self, u, action_idx, factor=0.8):
        """
        LATE feedback → decrease amplitude of the selected action peak.
        """
        u_new = u.copy()

        idx = self.action_bounds[action_idx]

        # Reduce amplitude locally
        u_new[idx] *= factor

        return u_new



    # def _make_later(self, u):
    #     # shift memory slightly righ
    #     #  as above but in opposite direction
    #     return np.roll(u, 20)

    def _make_later(self, u, ann, u_act_history):
        action_idx = self._action_index_from_time(
            ann.time_index, u_act_history
        )
        return self._make_later_action(u, action_idx)



    def _remove_peak(self, u, t_idx, history):
        # remove peak active at annotated time
        snapshot = history[t_idx]
        peak_pos = np.argmax(snapshot)
        u[peak_pos - 3 : peak_pos + 3] = -5.0
        return u


# TODO: edit actions!!!!!!


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

plot_fields = False
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
# -------- Feedback buffer -----------
# ====================================

annotation_buffer = AnnotationBuffer()

# Keep a copy of initial memory for learning
u_act_initial = u_act.copy()



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

    # ------------------------------------
    # Passive feedback collection (FAKE)
    # ------------------------------------
    # Later: replaced by ROS speech callback
    if i == 250:
        annotation_buffer.add(
            Annotation(
                time_index=i,
                feedback=FeedbackType.LATE
            )
        )





# ====================================
# -------- Post-run learning ---------
# ====================================

annotations = annotation_buffer.get_all()

if annotations:
    print(f"Applying {len(annotations)} feedback annotations")

    editor = MemoryEditor(x, input_positions)
    u_act_updated = editor.apply_feedback(
        u_act_initial=u_act_initial,
        u_act_history=u_act_history,
        annotations=annotations
    )

    # Save updated memory for next run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_path = results_dir / f"u_act_memory_updated_{timestamp}.npy"
    np.save(memory_path, u_act_updated)

    print(f"Updated memory saved to: {memory_path}")
else:
    print("No feedback annotations collected")




# ====================================
# -------- Memory comparison plot ----
# ====================================

plt.ioff()

fig, ax = plt.subplots(figsize=(10, 4))

for bucket_idx, color in zip(action_buckets, action_colors):

    # Initial memory
    ax.plot(
        x[bucket_idx],
        u_act_initial[bucket_idx],
        linestyle="--",
        linewidth=2,
        color=color,
    )

    # Updated memory
    ax.plot(
        x[bucket_idx],
        u_act_updated[bucket_idx],
        linestyle="-",
        linewidth=2,
        color=color,
    )

ax.set_xlabel("x")
ax.set_ylabel("Activity")
ax.set_title("Action memory: before vs after correction")

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=c, lw=2, label=f"Action {i+1}")
    for i, c in enumerate(action_colors)
]
legend_elements += [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Before"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="After"),
]

ax.legend(handles=legend_elements, loc="upper right")
ax.grid(True)

plt.tight_layout()
plt.show()

