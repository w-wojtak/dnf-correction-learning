# pylint: disable=C0200

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # type: ignore
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from utils import *
from dataclasses import dataclass
from typing import Optional
from enum import Enum

# ====================================
# -------- Human feedback ------------
# ====================================

class FeedbackType(Enum):
    EARLY = "early"
    LATE = "late"
    BEFORE = "before"
    AFTER = "after"
    SKIP = "skip"
    ADD = "add"
    SWAP = "swap"
    LOCK = "lock"


class ActionSelectionMode(Enum):
    TIME_BASED = "time"
    NAME_BASED = "name"
    AUTO = "auto"


@dataclass
class Annotation:
    """Flexible feedback annotation."""
    feedback: FeedbackType
    time_index: Optional[int] = None
    action_name: Optional[str] = None
    action_index: Optional[int] = None
    target_action: Optional[int] = None
    target_action_name: Optional[str] = None

@dataclass
class FeedbackFieldParams:
    kernel_type: str        # "gauss" or "osc"
    kernel_pars: list
    h_0: float              # resting level
    theta: float            # threshold
    tau: float              # time constant
    transient: bool         # True = decaying, False = sustained


class FeedbackField:
    def __init__(self, x, dx, params: FeedbackFieldParams, input_duration):
        self.x = x
        self.dx = dx
        self.params = params
        self.u = params.h_0 * np.ones(len(x))   # field state
        self.h = params.h_0 * np.ones(len(x))   # resting input
        self.s = np.zeros(len(x)) 
        self.input_duration = input_duration
        self._inject_step = None          # set when inject() is called
        
        # precompute kernel and its FFT
        if params.kernel_type == "gauss":
            self.kernel = kernel_gauss(x, *params.kernel_pars)
        elif params.kernel_type == "osc":
            self.kernel = kernel_osc(x, *params.kernel_pars)
        self.w_hat = np.fft.fft(self.kernel)

    def clear_input(self):
        self.s = np.zeros(len(self.x))
        self._inject_step = None
    
    def inject(self, center, amplitude, width, current_step):
        self.s = amplitude * np.exp(-0.5 * ((self.x - center) / width) ** 2)
        self._inject_step = current_step
    
    # def step(self, dt, current_step):
    #     if (self.params.transient and 
    #         self._inject_step is not None and 
    #         current_step >= self._inject_step + self.input_duration):
    #         self.clear_input()
    #     conv = compute_convolution(self.u, self.params.theta, self.w_hat)
    #     self.u += (dt / self.params.tau) * (-self.u + conv + self.h + self.s)
    
    def output(self):
        """Thresholded output."""
        return np.heaviside(self.u - self.params.theta, 1.0)


class AnnotationBuffer:
    """Passive buffer: stores human feedback during execution."""
    def __init__(self):
        self.annotations = []

    def add(self, annotation: Annotation):
        self.annotations.append(annotation)

    def get_all(self):
        return self.annotations


# ====================================
# -------- Initialization ------------
# ====================================

x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

input_positions = [-60, -30, 0, 30, 60]
input_indices = [np.argmin(np.abs(x - pos)) for pos in input_positions]

def compute_action_bounds(x, positions):
    """Compute spatial index ranges for each action bucket."""
    positions = np.array(positions)
    midpoints = (positions[:-1] + positions[1:]) / 2
    boundaries = np.concatenate(([x[0]], midpoints, [x[-1]]))
    return [np.where((x >= boundaries[i]) & (x < boundaries[i + 1]))[0]
            for i in range(len(positions))]

action_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ====================================
# -------- Memory editor -------------
# ====================================

class MemoryEditor:
    """Applies human feedback to modify initial u_act memory."""

    def __init__(self, x, action_centers, action_names=None,
                 default_mode=ActionSelectionMode.AUTO):
        self.x = x
        self.action_centers = np.array(action_centers)
        self.action_bounds = compute_action_bounds(x, action_centers)
        self.default_mode = default_mode

        if action_names is None:
            action_names = [f"action_{i}" for i in range(len(action_centers))]

        self.action_names = action_names
        self.name_to_index = {name.lower(): i for i, name in enumerate(action_names)}
        self.index_to_name = {i: name for i, name in enumerate(action_names)}

        self.h_amem_mask = np.ones(len(x))

        print(f"[MemoryEditor] Initialized with mode: {default_mode.value}")
        print(f"[MemoryEditor] Actions: {', '.join(action_names)}")

    def set_mode(self, mode):
        if isinstance(mode, str):
            mode = ActionSelectionMode(mode)
        self.default_mode = mode
        print(f"[MemoryEditor] Mode changed to: {mode.value}")

    def resolve_action_index(self, ann, u_act_history, mode=None):
        """Determine which action bucket to modify."""
        if mode is None:
            mode = self.default_mode

        if ann.action_index is not None:
            return ann.action_index, self.index_to_name[ann.action_index], "direct_index"

        if ann.action_name is not None:
            if mode in (ActionSelectionMode.NAME_BASED, ActionSelectionMode.AUTO):
                action_idx = self.get_action_index_from_name(ann.action_name)
                return action_idx, ann.action_name, "name"
            elif mode == ActionSelectionMode.TIME_BASED:
                print(f"[WARNING] Mode is TIME_BASED but action_name provided. "
                      f"Ignoring name '{ann.action_name}'")

        if ann.time_index is not None:
            if mode in (ActionSelectionMode.TIME_BASED, ActionSelectionMode.AUTO):
                action_idx = self._action_index_from_time(ann.time_index, u_act_history)
                return action_idx, self.index_to_name[action_idx], "time"
            elif mode == ActionSelectionMode.NAME_BASED:
                raise ValueError(
                    "Mode is NAME_BASED but only time_index provided. "
                    "Please provide action_name."
                )

        raise ValueError(
            f"Cannot determine action index. Annotation has:\n"
            f"  action_index: {ann.action_index}\n"
            f"  action_name: {ann.action_name}\n"
            f"  time_index: {ann.time_index}\n"
            f"At least one must be provided."
        )

    def resolve_target_action_index(self, ann, mode=None):
        """Resolve target action for SWAP operations."""
        if ann.target_action is not None:
            return ann.target_action, self.index_to_name[ann.target_action]

        if ann.target_action_name is not None:
            target_idx = self.get_action_index_from_name(ann.target_action_name)
            return target_idx, ann.target_action_name

        raise ValueError("SWAP feedback requires target_action or target_action_name")

    def apply_feedback(self, u_act_memory, u_act_initial, u_act_history,
                       theta_act, annotations, mode=None):
        """Apply feedback annotations with flexible action selection."""
        u_new = u_act_initial.copy()

        for i, ann in enumerate(annotations):
            print(f"\n[{i+1}/{len(annotations)}] Processing {ann.feedback.value} feedback")

            action_idx, action_name, method = self.resolve_action_index(
                ann, u_act_history, mode
            )
            print(f"  → Action: '{action_name}' (index {action_idx}) via {method}")

            if ann.feedback in (FeedbackType.LATE, FeedbackType.EARLY):
                u_new = self._apply_timing_feedback_direct(
                    u_act_memory, u_new, action_idx, ann.feedback, theta_act
                )

            elif ann.feedback == FeedbackType.SKIP:
                u_new = self._apply_skip_direct(u_new, action_idx)

            elif ann.feedback == FeedbackType.SWAP:
                target_idx, target_name = self.resolve_target_action_index(ann, mode)
                print(f"  → Target: '{target_name}' (index {target_idx})")
                u_new = self._apply_swap_direct(u_new, action_idx, target_idx)

            elif ann.feedback == FeedbackType.LOCK:
                self._apply_lock(action_idx)

        return u_new

    def get_lock_mask(self):
        return self.h_amem_mask.copy()

    # ========================================
    # Helper methods
    # ========================================

    def get_action_index_from_name(self, action_name):
        action_name_lower = action_name.lower().strip()
        if action_name_lower not in self.name_to_index:
            raise ValueError(
                f"Unknown action '{action_name}'. "
                f"Available actions: {', '.join(self.action_names)}"
            )
        return self.name_to_index[action_name_lower]

    def _action_index_from_time(self, time_index, u_act_history):
        return int(np.argmax(u_act_history[time_index]))

    def _apply_timing_feedback_direct(self, memory, u, action_idx, feedback_type,
                                      theta_act, factor=0.25):
        sign = -1 if feedback_type == FeedbackType.LATE else 1
        return self._modulate_action_amplitude(memory, u, action_idx, theta_act, factor, sign)

    def _apply_skip_direct(self, u, action_idx):
        """Skip/remove an action (mutates u in-place, no copy needed)."""
        print(f"[DEBUG] Skipping action {action_idx}")
        u[self.action_bounds[action_idx]] = u[0]
        return u

    def _apply_swap_direct(self, u, action_idx_a, action_idx_b):
        """Swap two actions (mutates u in-place, no copy needed)."""
        idx_a, idx_b = self.action_bounds[action_idx_a], self.action_bounds[action_idx_b]
        u[idx_a], u[idx_b] = u[idx_b].copy(), u[idx_a].copy()
        return u

    def _apply_lock(self, action_idx):
        """Lock an action by zeroing its mask region."""
        print(f"[DEBUG] Locking action {action_idx}")
        idx = self.action_bounds[action_idx]
        self.h_amem_mask[idx] = 0.0
        print(f"  → Mask set to 0 for {len(idx)} spatial points")

    def _modulate_action_amplitude(self, memory, u, action_idx, theta_act, factor, sign):
        """Peak-selective amplitude modulation."""
        idx = self.action_bounds[action_idx]
        u[idx] += sign * factor * np.heaviside(memory[idx] - theta_act, 1.0)
        return u


# ====================================
# -------- Project paths -------------
# ====================================

PROJECT_ROOT = Path.cwd()
results_dir = PROJECT_ROOT / "results_tuning"
results_dir.mkdir(exist_ok=True)

# ====================================
# -------- Load memory ---------------
# ====================================

folder = "/home/wwojtak/dnf_architecture_python/data_basic"
file1, ts1 = find_latest_file_with_prefix(folder, "u_field_1_")
file3, ts3 = find_latest_file_with_prefix(folder, "u_d_")

if ts1 != ts3:
    raise ValueError("Timestamps do not match across all files.")

u_field_1 = np.load(file1)
u_d = np.load(file3)

# ====================================
# --- Load fields for initialization --
# ====================================

try:
    u_d = u_d.flatten()
    h_d_initial = max(u_d)

    u_act = u_field_1 - h_d_initial + 1.5
    input_action_onset = u_field_1.flatten()
    h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

    u_sim = u_field_1.flatten() - h_d_initial + 1.5
    input_action_onset_2 = u_field_1.flatten()
    h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

    u_act_memory = u_field_1.copy()

except FileNotFoundError:
    print("No previous sequence memory found, initializing with default values.")
    u_act = np.zeros(np.shape(x))
    h_u_act = -3.2 * np.ones(np.shape(x))

# ====================================
# -------- Field parameters ----------
# ====================================

kernel_pars_act = [1.5, 0.8, 0.1]
kernel_pars_wm = [1.75, 0.5, 0.8]

h_0_wm = -1.0
theta_wm = 0.8

u_wm = h_0_wm * np.ones(np.shape(x))
h_u_wm = h_0_wm * np.ones(np.shape(x))

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

u_act_history = []
u_wm_history = []
h_u_amem = np.zeros(np.shape(x))

# ====================================
# -------- Online plotting -----------
# ====================================

plot_fields = False
plot_every = 5
plot_delay = 0.05

fig = axs = None
line_u_act = line_u_wm = None

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    line_u_act, = axs[0].plot(x, u_act, label="u_act(x)")
    axs[0].set_ylim(-5, 5)
    axs[0].set_ylabel("Activity")
    axs[0].legend()
    axs[0].set_title("Field u_act - Time = 0")

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
u_act_initial = u_act.copy()

# ====================================
# -------- Helper function -----------
# ====================================

def compute_convolution(u, theta, w_hat):
    """Compute FFT-based convolution for field dynamics."""
    f = np.heaviside(u - theta, 1)
    f_hat = np.fft.fft(f)
    return dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))


# -----

# input_duration = 50  # steps, same for all feedback fields

# # ── Transient field (SKIP) ─────────────────────────────────────────────
# skip_params = FeedbackFieldParams(
#     kernel_type="gauss",
#     kernel_pars=[1.0, 0.5, 0.05],   # same format as kernel_pars_act
#     h_0=-1.0,                        # strongly sub-threshold at rest
#     theta=0.5,
#     tau=1.0,                         # fast decay → transient
#     transient=True
# )

# # ── Sustained field (LOCK) ─────────────────────────────────────────────
# lock_params = FeedbackFieldParams(
#     kernel_type="gauss",
#     kernel_pars=[2.0, 0.8, 0.05],   # stronger recurrent excitation → self-sustaining
#     h_0=-1.0,
#     theta=0.5,
#     tau=10.0,                        # slower → sustained
#     transient=False
# )

# skip_field = FeedbackField(x, dx, skip_params, input_duration)
# lock_field = FeedbackField(x, dx, lock_params, input_duration=50)

# # ── Inject and run ─────────────────────────────────────────────────────
# # skip_field.inject(center=0, amplitude=3.0, width=5.0, current_step=i)
# # lock_field.inject(center=0, amplitude=3.0, width=5.0, current_step=i)

# skip_history = []
# lock_history = []


# inject_at = 10  # step at which input is triggered
# lock_history = []

# # for i in range(200):
# #     if i == inject_at:
# #         skip_field.inject(center=0, amplitude=3.0, width=5.0, current_step=i)
    
# #     skip_field.step(dt, current_step=i)
# #     skip_history.append(skip_field.u.copy())

# for i in range(200):
#     if i == inject_at:
#         lock_field.inject(center=0, amplitude=3.0, width=5.0, current_step=i)
    
#     lock_field.step(dt, current_step=i)
#     lock_history.append(lock_field.u.copy())


# fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# axs[0].plot(x, lock_history[0],   label="t=0")
# axs[0].plot(x, lock_history[50],  label="t=50")
# axs[0].plot(x, lock_history[-1],  label="t=end")
# axs[0].axhline(lock_params.theta, color='k', linestyle='--', label='theta')
# axs[0].set_title("LOCK field - u(x)")
# axs[0].legend()

# axs[1].plot(x, lock_field.output())
# axs[1].set_title("LOCK field - output at t=end")

# plt.tight_layout()
# plt.show()

# fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# axs[0].plot(x, skip_history[0],   label="t=0")
# axs[0].plot(x, skip_history[50],  label="t=50")
# axs[0].plot(x, skip_history[65],  label="t=65")
# axs[0].plot(x, skip_history[-1],  label="t=end")
# axs[0].axhline(skip_params.theta, color='k', linestyle='--', label='theta')
# axs[0].set_title("SKIP field - u(x)")
# axs[0].legend()

# axs[1].plot(x, skip_field.output())
# axs[1].set_title("SKIP field - output at t=end")

# plt.tight_layout()
# plt.show()

# # ── Plot final state ───────────────────────────────────────────────────
# fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

# axs[0, 0].plot(x, skip_history[0],   label="t=0")
# axs[0, 0].plot(x, skip_history[50],  label="t=50")
# axs[0, 0].plot(x, skip_history[-1],  label="t=end")
# axs[0, 0].axhline(skip_params.theta, color='k', linestyle='--', label='theta')
# axs[0, 0].set_title("SKIP field - u(x)")
# axs[0, 0].legend()

# axs[0, 1].plot(x, skip_field.output())
# axs[0, 1].set_title("SKIP field - output at t=end")

# axs[1, 0].plot(x, lock_history[0],   label="t=0")
# axs[1, 0].plot(x, lock_history[50],  label="t=50")
# axs[1, 0].plot(x, lock_history[-1],  label="t=end")
# axs[1, 0].axhline(lock_params.theta, color='k', linestyle='--', label='theta')
# axs[1, 0].set_title("LOCK field - u(x)")
# axs[1, 0].legend()

# axs[1, 1].plot(x, lock_field.output())
# axs[1, 1].set_title("LOCK field - output at t=end")

# plt.tight_layout()
# plt.show()

# -----


# ── Feedback field params ─────────────────────────────────────────────
input_duration = 50  # steps, same for all feedback fields

skip_params = FeedbackFieldParams(
    kernel_type="gauss",
    kernel_pars=[1.0, 0.5, 0.05],
    h_0=-1.0,
    theta=0.5,
    tau=1.0,
    transient=True
)

lock_params = FeedbackFieldParams(
    kernel_type="gauss",
    kernel_pars=[1.5, 0.75, 0.05],
    h_0=-1.0,
    theta=0.5,
    tau=1.0,
    transient=False
)

feedback_fields = {
    FeedbackType.SKIP: FeedbackField(x, dx, skip_params, input_duration),
    FeedbackType.LOCK: FeedbackField(x, dx, lock_params, input_duration),
}

# ── Feedback field history ────────────────────────────────────────────
skip_history = []
lock_history = []

# ====================================
# -------- Recall simulation ---------
# ====================================

# todo: inputs for tuning ????
# action name + what ??? we need action and input_position.....
# action name + object ?? from vision ???? 
# better than from time step, because it would need to be when the action is executed??
# but then why do we run u_act and u_wm ??
# BUT objects are in robot's hands during execution....
# both?? or sth else ? 
# or just correct at the end??
# but then user might already forget
# OR different ways depending on the action ???
# to justify running u_act and u_wm....
# so how to correct during the run but affect u_amem only at the end?????
# 



for i in range(len(t)):
    # ── main fields ──────────────────────────────────────────────────
    conv_act = compute_convolution(u_act, theta_act, w_hat_act)
    conv_wm  = compute_convolution(u_wm, theta_wm, w_hat_wm)
    f_act = np.heaviside(u_act - theta_act, 1)
    f_wm  = np.heaviside(u_wm - theta_wm, 1)

    h_u_act += dt / tau_h_act

    u_act += dt * (-u_act + conv_act + input_action_onset + h_u_act - 6.0 * f_wm * conv_wm)
    u_wm  += (dt / 1.25) * (-u_wm + conv_wm + 8 * (f_act * u_act) + h_u_wm)

    # ── feedback fields ───────────────────────────────────────────────
    for ftype, ff in feedback_fields.items():
        if (ff.params.transient and
            ff._inject_step is not None and
            i >= ff._inject_step + ff.input_duration):
            ff.clear_input()

        conv_ff = compute_convolution(ff.u, ff.params.theta, ff.w_hat)
        ff.u += (dt / ff.params.tau) * (-ff.u + conv_ff + ff.h + ff.s)

    # ── history ───────────────────────────────────────────────────────
    u_act_history.append([u_act[idx] for idx in input_indices])
    u_wm_history.append([u_wm[idx] for idx in input_indices])
    skip_history.append(feedback_fields[FeedbackType.SKIP].u.copy())
    lock_history.append(feedback_fields[FeedbackType.LOCK].u.copy())

    # ── feedback injection example ────────────────────────────────────
    if i == 250:
        feedback_fields[FeedbackType.SKIP].inject(
            center=input_positions[1],   # grasp
            amplitude=3.0, width=5.0, current_step=i
        )
    if i == 500:
        feedback_fields[FeedbackType.LOCK].inject(
            center=input_positions[1],   # grasp
            amplitude=3.0, width=5.0, current_step=i
        )



skip_history = np.array(skip_history)
lock_history = np.array(lock_history)

fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=False)

# ── SKIP: spatial snapshot at key times ──────────────────────────────
axs[0, 0].plot(x, skip_history[0],    label="t=0")
axs[0, 0].plot(x, skip_history[250],  label="t=inject")
axs[0, 0].plot(x, skip_history[300],  label="t=inject+50")
axs[0, 0].plot(x, skip_history[-1],   label="t=end")
axs[0, 0].axhline(skip_params.theta, color='k', linestyle='--', label='theta')
axs[0, 0].axvline(input_positions[1], color='gray', linestyle=':', label='target')
axs[0, 0].set_title("SKIP field - u(x) snapshots")
axs[0, 0].set_xlabel("x")
axs[0, 0].legend()

# ── SKIP: activity over time at target location ───────────────────────
target_idx = np.argmin(np.abs(x - input_positions[1]))
axs[0, 1].plot(skip_history[:, target_idx])
axs[0, 1].axhline(skip_params.theta, color='k', linestyle='--', label='theta')
axs[0, 1].axvline(250, color='r', linestyle='--', label='inject')
axs[0, 1].set_title("SKIP field - u(target) over time")
axs[0, 1].set_xlabel("time step")
axs[0, 1].legend()

# ── LOCK: spatial snapshot at key times ──────────────────────────────
axs[1, 0].plot(x, lock_history[0],    label="t=0")
axs[1, 0].plot(x, lock_history[500],  label="t=inject")
axs[1, 0].plot(x, lock_history[550],  label="t=inject+50")
axs[1, 0].plot(x, lock_history[-1],   label="t=end")
axs[1, 0].axhline(lock_params.theta, color='k', linestyle='--', label='theta')
axs[1, 0].axvline(input_positions[1], color='gray', linestyle=':', label='target')
axs[1, 0].set_title("LOCK field - u(x) snapshots")
axs[1, 0].set_xlabel("x")
axs[1, 0].legend()

# ── LOCK: activity over time at target location ───────────────────────
axs[1, 1].plot(lock_history[:, target_idx])
axs[1, 1].axhline(lock_params.theta, color='k', linestyle='--', label='theta')
axs[1, 1].axvline(500, color='r', linestyle='--', label='inject')
axs[1, 1].set_title("LOCK field - u(target) over time")
axs[1, 1].set_xlabel("time step")
axs[1, 1].legend()

plt.tight_layout()
plt.show()


# for i in range(len(t)):
#     conv_act = compute_convolution(u_act, theta_act, w_hat_act)
#     conv_wm = compute_convolution(u_wm, theta_wm, w_hat_wm)
#     f_act = np.heaviside(u_act - theta_act, 1)
#     f_wm = np.heaviside(u_wm - theta_wm, 1)

#     h_u_act += dt / tau_h_act

#     u_act += dt * (-u_act + conv_act + input_action_onset + h_u_act - 6.0 * f_wm * conv_wm)
#     u_wm += (dt / 1.25) * (-u_wm + conv_wm + 8 * (f_act * u_act) + h_u_wm)

#     u_act_history.append([u_act[idx] for idx in input_indices])
#     u_wm_history.append([u_wm[idx] for idx in input_indices])

#     if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
#         line_u_act.set_ydata(u_act)
#         line_u_wm.set_ydata(u_wm)
#         axs[0].set_title(f"Field u_act - Time = {i}")
#         axs[1].set_title(f"Field u_wm - Time = {i}")
#         plt.pause(plot_delay)

#     # ====================================
#     # Passive feedback collection - EXAMPLES
#     # ====================================

#     # Example: SWAP feedback
#     if i == 250:
#         annotation_buffer.add(
#             Annotation(
#                 feedback=FeedbackType.SWAP,
#                 action_name="grasp",
#                 target_action_name="transport"
#             )
#         )

#     # # Example: LOCK feedback
#     # if i == 250:
#     #     annotation_buffer.add(
#     #         Annotation(
#     #             feedback=FeedbackType.LOCK,
#     #             action_name="grasp"
#     #         )
#     #     )

# ====================================
# -------- Post-run learning ---------
# ====================================

annotations = annotation_buffer.get_all()

# Default outputs (no feedback case)
u_act_updated = u_act_initial.copy()
h_amem_mask = np.ones(len(x))

if annotations:
    print(f"\nApplying {len(annotations)} feedback annotations")

    editor = MemoryEditor(
        x,
        input_positions,
        action_names=["reach", "grasp", "lift", "transport", "place"],
        default_mode=ActionSelectionMode.AUTO
    )

    print("\nMemory peak per action:")
    for k, idx in enumerate(editor.action_bounds):
        print(f"  {editor.action_names[k]}: {np.max(u_act_memory[idx]):.3f}")

    u_act_updated = editor.apply_feedback(
        u_act_memory=u_act_memory,
        u_act_initial=u_act_initial,
        u_act_history=u_act_history,
        theta_act=theta_act,
        annotations=annotations
    )

    print("\n[DEBUG] Action peak changes (Δ):")
    for k, idx in enumerate(editor.action_bounds):
        delta = np.max(u_act_updated[idx]) - np.max(u_act_initial[idx])
        print(f"  {editor.action_names[k]}: Δpeak = {delta:.3f}")

    h_amem_mask = editor.get_lock_mask()

    # Save updated memory and mask
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_path = results_dir / f"u_act_memory_updated_{timestamp}.npy"
    mask_path = results_dir / f"h_amem_mask_{timestamp}.npy"

    np.save(memory_path, u_act_updated)
    np.save(mask_path, h_amem_mask)

    print(f"\nUpdated memory saved to: {memory_path}")
    print(f"Lock mask saved to: {mask_path}")

    locked_regions = np.sum(h_amem_mask == 0)
    total_points = len(h_amem_mask)
    print(f"\nLock mask summary:")
    print(f"  Locked points: {locked_regions}/{total_points} ({100*locked_regions/total_points:.1f}%)")
    print(f"  Unlocked points: {total_points - locked_regions}/{total_points} ({100*(total_points-locked_regions)/total_points:.1f}%)")

else:
    print("No feedback annotations collected")

# ====================================
# -------- Memory comparison plot ----
# ====================================

# Use editor's action_bounds if available, otherwise compute fresh
action_buckets = editor.action_bounds if annotations else compute_action_bounds(x, input_positions)
action_names_plot = editor.action_names if annotations else [f"action_{i}" for i in range(len(input_positions))]

plt.ioff()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for bucket_idx, color in zip(action_buckets, action_colors):
    ax1.plot(x[bucket_idx], u_act_initial[bucket_idx], linestyle="--", linewidth=2, color=color)
    ax1.plot(x[bucket_idx], u_act_updated[bucket_idx], linestyle="-", linewidth=2, color=color)

ax1.set_ylabel("Activity")
ax1.set_title("Action memory: before vs after correction")

legend_elements = [Line2D([0], [0], color=c, lw=2, label=action_names_plot[i])
                   for i, c in enumerate(action_colors)]
legend_elements += [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Before"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="After"),
]
ax1.legend(handles=legend_elements, loc="upper right")
ax1.grid(True)

ax2.fill_between(x, 0, h_amem_mask, alpha=0.3, color='gray', label='Unlocked (1)')
ax2.fill_between(x, 0, 1 - h_amem_mask, alpha=0.5, color='red', label='Locked (0)')
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlabel("x")
ax2.set_ylabel("Mask value")
ax2.set_title("Lock mask (h_amem_mask)")
ax2.legend(loc="upper right")
ax2.grid(True)

for pos in input_positions:
    ax1.axvline(pos, color='gray', linestyle=':', alpha=0.3)
    ax2.axvline(pos, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()