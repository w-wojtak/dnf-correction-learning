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
# NOTE: Must be defined BEFORE Annotation class!

class FeedbackType(Enum):
    EARLY = "early"
    LATE = "late"
    BEFORE = "before"
    AFTER = "after"
    SKIP = "skip"
    ADD = "add"
    SWAP = "swap"


class ActionSelectionMode(Enum):
    """How to determine which action to modify."""
    TIME_BASED = "time"      # Infer from when feedback occurred
    NAME_BASED = "name"      # User specifies action name
    AUTO = "auto"            # Try name first, fall back to time


@dataclass
class Annotation:
    """
    Flexible feedback annotation.
    
    Supports multiple ways to specify which action:
    - time_index: auto-detect from neural activity at that time
    - action_name: user explicitly names the action
    - action_index: directly specify the bucket index (advanced use)
    """
    feedback: FeedbackType
    time_index: Optional[int] = None
    action_name: Optional[str] = None
    action_index: Optional[int] = None  # Direct index specification
    target_action: Optional[int] = None  # For SWAP (numeric)
    target_action_name: Optional[str] = None  # For SWAP (by name)


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

# Compute action buckets
def compute_action_bounds(x, positions):
    """Compute spatial index ranges for each action bucket."""
    positions = np.array(positions)
    midpoints = (positions[:-1] + positions[1:]) / 2
    boundaries = np.concatenate(([x[0]], midpoints, [x[-1]]))
    
    return [np.where((x >= boundaries[i]) & (x < boundaries[i + 1]))[0] 
            for i in range(len(positions))]

action_buckets = compute_action_bounds(x, input_positions)
action_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ====================================
# -------- Memory editor -------------
# ====================================

class MemoryEditor:
    """Applies human feedback to modify initial u_act memory."""

    def __init__(self, x, action_centers, action_names=None, 
                 default_mode=ActionSelectionMode.AUTO):
        """
        Initialize memory editor.
        
        Args:
            x: Spatial array
            action_centers: List of action center positions
            action_names: List of human-readable action names (optional)
            default_mode: Default action selection strategy
        """
        self.x = x
        self.action_centers = np.array(action_centers)
        self.action_bounds = compute_action_bounds(x, action_centers)
        self.default_mode = default_mode
        
        # Create action name mappings
        if action_names is None:
            action_names = [f"action_{i}" for i in range(len(action_centers))]
        
        self.action_names = action_names
        self.name_to_index = {name.lower(): i for i, name in enumerate(action_names)}
        self.index_to_name = {i: name for i, name in enumerate(action_names)}
        
        print(f"[MemoryEditor] Initialized with mode: {default_mode.value}")
        print(f"[MemoryEditor] Actions: {', '.join(action_names)}")
    
    def set_mode(self, mode):
        """Change the default action selection mode."""
        if isinstance(mode, str):
            mode = ActionSelectionMode(mode)
        self.default_mode = mode
        print(f"[MemoryEditor] Mode changed to: {mode.value}")
    
    def resolve_action_index(self, ann, u_act_history, mode=None):
        """
        Determine which action bucket to modify.
        
        Priority order (can be controlled by mode):
        1. action_index (if explicitly provided)
        2. action_name (if provided and mode allows)
        3. time_index (if provided and mode allows)
        
        Args:
            ann: Annotation object
            u_act_history: History of action activations
            mode: Override default mode for this annotation
        
        Returns:
            tuple: (action_index, action_name, method_used)
        """
        if mode is None:
            mode = self.default_mode
        
        # 1. Direct index specification (highest priority)
        if ann.action_index is not None:
            action_name = self.index_to_name[ann.action_index]
            return ann.action_index, action_name, "direct_index"
        
        # 2. Name-based selection
        if ann.action_name is not None:
            if mode in (ActionSelectionMode.NAME_BASED, ActionSelectionMode.AUTO):
                action_idx = self.get_action_index_from_name(ann.action_name)
                return action_idx, ann.action_name, "name"
            elif mode == ActionSelectionMode.TIME_BASED:
                print(f"[WARNING] Mode is TIME_BASED but action_name provided. "
                      f"Ignoring name '{ann.action_name}'")
        
        # 3. Time-based selection
        if ann.time_index is not None:
            if mode in (ActionSelectionMode.TIME_BASED, ActionSelectionMode.AUTO):
                action_idx = self._action_index_from_time(ann.time_index, u_act_history)
                action_name = self.index_to_name[action_idx]
                return action_idx, action_name, "time"
            elif mode == ActionSelectionMode.NAME_BASED:
                raise ValueError(
                    "Mode is NAME_BASED but only time_index provided. "
                    "Please provide action_name."
                )
        
        # If we get here, insufficient information
        raise ValueError(
            f"Cannot determine action index. Annotation has:\n"
            f"  action_index: {ann.action_index}\n"
            f"  action_name: {ann.action_name}\n"
            f"  time_index: {ann.time_index}\n"
            f"At least one must be provided."
        )
    
    def resolve_target_action_index(self, ann, mode=None):
        """
        Resolve target action for SWAP operations.
        Similar to resolve_action_index but for swap targets.
        """
        if mode is None:
            mode = self.default_mode
        
        # Direct index
        if ann.target_action is not None:
            return ann.target_action, self.index_to_name[ann.target_action]
        
        # By name
        if ann.target_action_name is not None:
            target_idx = self.get_action_index_from_name(ann.target_action_name)
            return target_idx, ann.target_action_name
        
        raise ValueError("SWAP feedback requires target_action or target_action_name")
    
    def apply_feedback(self, u_act_memory, u_act_initial, u_act_history, 
                      theta_act, annotations, mode=None):
        """
        Apply feedback annotations with flexible action selection.
        
        Args:
            u_act_memory: Memory field (for computing modulations)
            u_act_initial: Initial activation to modify
            u_act_history: Time series of activations
            theta_act: Threshold for action detection
            annotations: List of Annotation objects
            mode: Override default mode for all annotations (optional)
        
        Returns:
            Modified activation field
        """
        u_new = u_act_initial.copy()

        for i, ann in enumerate(annotations):
            print(f"\n[{i+1}/{len(annotations)}] Processing {ann.feedback.value} feedback")
            
            # Resolve which action to modify
            action_idx, action_name, method = self.resolve_action_index(
                ann, u_act_history, mode
            )
            print(f"  → Action: '{action_name}' (index {action_idx}) via {method}")
            
            # Apply the appropriate feedback
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

        return u_new
    
    # ========================================
    # Helper methods
    # ========================================
    
    def get_action_index_from_name(self, action_name):
        """Convert action name to action index."""
        action_name_lower = action_name.lower().strip()
        
        if action_name_lower not in self.name_to_index:
            available = ", ".join(self.action_names)
            raise ValueError(
                f"Unknown action '{action_name}'. "
                f"Available actions: {available}"
            )
        
        return self.name_to_index[action_name_lower]
    
    def _action_index_from_time(self, time_index, u_act_history):
        """Return which action bucket was active at this time."""
        values = u_act_history[time_index]
        action_idx = int(np.argmax(values))
        return action_idx
    
    def _apply_timing_feedback_direct(self, memory, u, action_idx, feedback_type, 
                                     theta_act, factor=0.25):
        """Apply timing feedback to a specific action."""
        sign = -1 if feedback_type == FeedbackType.LATE else 1
        return self._modulate_action_amplitude(memory, u, action_idx, theta_act, factor, sign)
    
    def _apply_skip_direct(self, u, action_idx):
        """Skip/remove an action."""
        print(f"[DEBUG] Skipping action {action_idx}")
        u_new = u.copy()
        u_new[self.action_bounds[action_idx]] = u_new[0]
        return u_new
    
    def _apply_swap_direct(self, u, action_idx_a, action_idx_b):
        """Swap two actions."""
        u_new = u.copy()
        idx_a, idx_b = self.action_bounds[action_idx_a], self.action_bounds[action_idx_b]
        u_new[idx_a], u_new[idx_b] = u_new[idx_b].copy(), u_new[idx_a].copy()
        return u_new
    
    def _modulate_action_amplitude(self, memory, u, action_idx, theta_act, factor, sign):
        """Generic peak-selective amplitude modulation."""
        u_new = u.copy()
        idx = self.action_bounds[action_idx]
        f = np.heaviside(memory[idx] - theta_act, 1.0)
        u_new[idx] += sign * factor * f
        return u_new


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

# ====================================
# -------- Recall simulation ---------
# ====================================

for i in range(len(t)):
    # Compute convolutions
    conv_act = compute_convolution(u_act, theta_act, w_hat_act)
    conv_wm = compute_convolution(u_wm, theta_wm, w_hat_wm)
    f_act = np.heaviside(u_act - theta_act, 1)
    f_wm = np.heaviside(u_wm - theta_wm, 1)

    # Update field states
    h_u_act += dt / tau_h_act

    u_act += dt * (-u_act + conv_act + input_action_onset + h_u_act - 6.0 * f_wm * conv_wm)
    u_wm += (dt / 1.25) * (-u_wm + conv_wm + 8 * (f_act * u_act) + h_u_wm)

    # Store history
    u_act_history.append([u_act[idx] for idx in input_indices])
    u_wm_history.append([u_wm[idx] for idx in input_indices])

    # Online plot update
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        line_u_act.set_ydata(u_act)
        line_u_wm.set_ydata(u_wm)
        axs[0].set_title(f"Field u_act - Time = {i}")
        axs[1].set_title(f"Field u_wm - Time = {i}")
        plt.pause(plot_delay)

    # ====================================
    # Passive feedback collection
    # ====================================
    # CHOOSE ONE OF THE FOLLOWING EXAMPLES:
    
    # Example 1: Time-based (original)
    # if i == 250:
    #     annotation_buffer.add(
    #         Annotation(
    #             feedback=FeedbackType.SWAP,
    #             time_index=i,
    #             target_action=3
    #         )
    #     )
    
    # Example 2: Name-based
    if i == 250:
        annotation_buffer.add(
            Annotation(
                feedback=FeedbackType.SKIP,
                action_name="reach",  # ✅ This is index 1
                target_action_name="transport"  # ✅ This is index 3
            )
        )
# ====================================
# -------- Post-run learning ---------
# ====================================

annotations = annotation_buffer.get_all()

if annotations:
    print(f"\nApplying {len(annotations)} feedback annotations")

    # Initialize editor with action names and mode
    editor = MemoryEditor(
        x, 
        input_positions,
        action_names=["reach", "grasp", "lift", "transport", "place"],
        default_mode=ActionSelectionMode.AUTO  # Can switch to TIME_BASED or NAME_BASED
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

    # Save updated memory for next run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_path = results_dir / f"u_act_memory_updated_{timestamp}.npy"
    np.save(memory_path, u_act_updated)

    print(f"\nUpdated memory saved to: {memory_path}")
else:
    print("No feedback annotations collected")

# ====================================
# -------- Memory comparison plot ----
# ====================================

plt.ioff()
fig, ax = plt.subplots(figsize=(10, 4))

for bucket_idx, color in zip(action_buckets, action_colors):
    ax.plot(x[bucket_idx], u_act_initial[bucket_idx], linestyle="--", linewidth=2, color=color)
    ax.plot(x[bucket_idx], u_act_updated[bucket_idx], linestyle="-", linewidth=2, color=color)

ax.set_xlabel("x")
ax.set_ylabel("Activity")
ax.set_title("Action memory: before vs after correction")

# Custom legend
legend_elements = [Line2D([0], [0], color=c, lw=2, label=f"{editor.action_names[i]}") 
                   for i, c in enumerate(action_colors)]
legend_elements += [
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Before"),
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="After"),
]

ax.legend(handles=legend_elements, loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.show()