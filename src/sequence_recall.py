# pylint: disable=C0200
from scipy.ndimage import gaussian_filter1d  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from datetime import datetime
from utils import *
from pathlib import Path
import sys

# ====================================
# -------- Project Setup -------------
# ====================================

PROJECT_ROOT = Path.cwd()
results_dir = PROJECT_ROOT / "results_recall"
results_dir.mkdir(exist_ok=True)

# Trial configuration
trial_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1
input_onset_time_by_trial = {
    1: [8, 19, 28, 38, 48],
    2: [8, 19, 28, 38, 48],
    3: [8, 19, 28, 38, 48],
}
input_onset_time_2 = input_onset_time_by_trial.get(trial_number, input_onset_time_by_trial[1])

# ====================================
# -------- Parameters ----------------
# ====================================

# Kernel parameters
kernel_pars_act = [1.5, 0.8, 0.1]
kernel_pars_sim = [1.7, 0.8, 0.7]
kernel_pars_wm = [1.75, 0.5, 0.8]
kernel_pars_f = [1.5, 0.8, 0.0]
kernel_pars_error = [1.5, 0.8, 0.0]
kernel_pars_inh = [3, 1.5, 0.0]

# Spatial and temporal parameters
x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

# Adaptation and plotting
beta_adapt = 0.001
plot_fields = False
plot_every = 5
plot_delay = 0.05

# ====================================
# -------- Load Memory ---------------
# ====================================

folder = PROJECT_ROOT / "results_learning"
file1, ts1 = find_latest_file_with_prefix(folder, "u_field_1_")
file3, ts3 = find_latest_file_with_prefix(folder, "u_d_")

if ts1 != ts3:
    raise ValueError("Timestamps do not match across all files.")

u_field_1 = np.load(file1)
u_d = np.load(file3)

# ====================================
# -------- Input Configuration -------
# ====================================

input_flag = True
input_shape = [3, 1.5]
input_duration = [5, 5, 5, 5, 5]
input_position_1 = [-60, -30, 0, 30, 60]
input_onset_time_1 = [3, 8, 12, 16, 20]
input_position_2 = input_position_1

input_pars_1 = [input_shape, input_position_1, input_onset_time_1, input_duration]
input_pars_2 = [input_shape, input_position_2, input_onset_time_2, input_duration]

inputs_1 = get_inputs(x, t, dt, input_pars_1, input_flag)
inputs_2 = get_inputs(x, t, dt, input_pars_2, input_flag)
input_agent_robot_feedback = np.zeros((len(t), len(x)))

# ====================================
# -------- Field Initialization ------
# ====================================

try:
    u_d = u_d.flatten()
    h_d_initial = max(u_d)

    if trial_number == 1:
        # First trial initialization
        u_act = u_field_1 - h_d_initial + 1.5
        input_action_onset = u_field_1.flatten()
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

        u_sim = u_field_1.flatten() - h_d_initial + 1.5
        input_action_onset_2 = u_field_1.flatten()
        h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5
    else:
        # Load adaptation memory from previous trial
        print(f"Loading h_amem from {folder}")
        latest_h_amem_file, _ = find_latest_file_with_prefix(folder, "h_u_amem_")
        latest_h_amem = np.load(latest_h_amem_file, allow_pickle=True)

        u_act = u_field_1.flatten() - h_d_initial + 1.5 - latest_h_amem
        input_action_onset = u_field_1.flatten() - latest_h_amem
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

        u_sim = u_field_1.flatten() - h_d_initial + 1.5
        input_action_onset_2 = u_field_1.flatten()
        h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

except FileNotFoundError:
    print("No previous sequence memory found, initializing with default values.")
    u_act = np.zeros(np.shape(x))
    h_u_act = -3.2 * np.ones(np.shape(x))

# Working memory parameters
h_0_wm = -1.0
theta_wm = 0.8
u_wm = h_0_wm * np.ones(np.shape(x))
h_u_wm = h_0_wm * np.ones(np.shape(x))

# Thresholds and time constants
tau_h_act = 20
theta_act = 1.5
tau_h_sim = 10
theta_sim = 1.5
theta_error = 1.5

# Compute kernels and FFTs
kernel_act = kernel_gauss(x, *kernel_pars_act)
kernel_sim = kernel_gauss(x, *kernel_pars_sim)
kernel_wm = kernel_osc(x, *kernel_pars_wm)
kernel_inh = kernel_gauss(x, *kernel_pars_inh)

w_hat_act = np.fft.fft(kernel_act)
w_hat_sim = np.fft.fft(kernel_sim)
w_hat_wm = np.fft.fft(kernel_wm)
w_hat_inh = np.fft.fft(kernel_inh)

# Feedback fields
h_f = -1.0
w_hat_f = w_hat_act
tau_h_f = tau_h_act
theta_f = theta_act

u_f1 = h_f * np.ones(np.shape(x))
u_f2 = h_f * np.ones(np.shape(x))
u_error = h_f * np.ones(np.shape(x))

# Field histories
u_act_history = []
u_sim_history = []
u_wm_history = []
u_f1_history = []
u_f2_history = []

# Adaptation memory field
h_u_amem = np.zeros(np.shape(x))

# ====================================
# -------- Plotting Setup ------------
# ====================================

fig = axs = line1_field = line2_field = line3_field = line4_field = line5_field = None

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(14, 14), sharex=True)

    # Setup subplots
    fields = [
        (u_act, 'u_act', 0, 0),
        (u_sim, 'u_sim', 0, 1),
        (u_f1, 'u_f1', 1, 0),
        (u_f2, 'u_f2', 1, 1),
        (u_wm, 'u_wm', 2, 0),
    ]
    
    lines = []
    for field, name, row, col in fields:
        line, = axs[row, col].plot(x, field, label=f'{name}(x)')
        axs[row, col].set_ylim(-5, 5)
        axs[row, col].set_ylabel("Activity")
        axs[row, col].legend()
        axs[row, col].set_title(f"Field {name} - Time = 0")
        lines.append(line)
    
    line1_field, line2_field, line3_field, line4_field, line5_field = lines
    axs[2, 0].set_xlabel("x")
    axs[2, 1].axis("off")
    plt.tight_layout()

# ====================================
# -------- Gaussian Input Setup ------
# ====================================

input_positions = input_position_1
input_indices = [np.argmin(np.abs(x - pos)) for pos in input_positions]
threshold_crossed = {pos: False for pos in input_positions}

gaussian_amplitude = 3
gaussian_width = 1.5

def gaussian_input(x, center, amplitude, width):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

# ====================================
# -------- Helper Function -----------
# ====================================

def compute_convolution(u, theta, w_hat):
    """Compute FFT-based convolution for field dynamics."""
    f = np.heaviside(u - theta, 1)
    f_hat = np.fft.fft(f)
    return dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))

# ====================================
# -------- Main Simulation Loop ------
# ====================================

for i in range(len(t)):
    input_agent_2 = inputs_2[i, :]

    # Compute all convolutions
    f_f1 = np.heaviside(u_f1 - theta_f, 1)
    f_f2 = np.heaviside(u_f2 - theta_f, 1)
    f_act = np.heaviside(u_act - theta_act, 1)
    f_sim = np.heaviside(u_sim - theta_sim, 1)
    f_wm = np.heaviside(u_wm - theta_wm, 1)
    f_error = np.heaviside(u_error - theta_error, 1)

    conv_f1 = compute_convolution(u_f1, theta_f, w_hat_f)
    conv_f2 = compute_convolution(u_f2, theta_f, w_hat_f)
    conv_act = compute_convolution(u_act, theta_act, w_hat_act)
    conv_sim = compute_convolution(u_sim, theta_sim, w_hat_sim)
    conv_wm = compute_convolution(u_wm, theta_wm, w_hat_wm)
    conv_inh = dx * np.fft.ifftshift(np.real(np.fft.ifft(np.fft.fft(f_wm) * w_hat_inh)))
    conv_error = compute_convolution(u_error, theta_error, w_hat_act)

    # Update field states
    h_u_act += dt / tau_h_act
    h_u_sim += dt / tau_h_sim

    u_act += dt * (-u_act + conv_act + input_action_onset + h_u_act - 6.0 * f_wm * conv_wm)
    u_sim += dt * (-u_sim + conv_sim + input_action_onset_2 + h_u_sim - 6.0 * f_wm * conv_wm)
    u_wm += (dt/1.25) * (-u_wm + conv_wm + 8 * ((f_f1 * u_f1) * (f_f2 * u_f2)) + h_u_wm)
    u_f1 += dt * (-u_f1 + conv_f1 + input_agent_robot_feedback[i, :] + h_f - 1 * f_wm * conv_wm)
    u_f2 += dt * (-u_f2 + conv_f2 + input_agent_2 + h_f - 1 * f_wm * conv_wm)
    u_error += dt * (-u_error + conv_error + h_f - 2 * f_sim * conv_sim)

    # Adaptation update
    h_u_amem += beta_adapt * (1 - (f_f2 * f_f1)) * (f_f1 - f_f2)

    # Store history
    u_act_history.append([u_act[idx] for idx in input_indices])
    u_sim_history.append([u_sim[idx] for idx in input_indices])
    u_wm_history.append([u_wm[idx] for idx in input_indices])
    u_f1_history.append([u_f1[idx] for idx in input_indices])
    u_f2_history.append([u_f2[idx] for idx in input_indices])

    # Detect threshold crossing and add robot feedback
    for idx, pos in zip(input_indices, input_positions):
        if not threshold_crossed[pos] and u_act[idx] > theta_act:
            print(f"Threshold crossed at position {pos} and time {i*dt}")
            threshold_crossed[pos] = True
            
            time_on = i + 20
            time_off = len(t)
            gaussian = gaussian_amplitude * np.exp(-((x - pos) ** 2) / (2 * gaussian_width ** 2))
            input_agent_robot_feedback[time_on:time_off, :] += gaussian

    # Update plots
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        line1_field.set_ydata(u_act)
        line2_field.set_ydata(u_sim)
        line3_field.set_ydata(u_f1)
        line4_field.set_ydata(u_f2)
        line5_field.set_ydata(u_wm)

        axs[0, 0].set_title(f"Field u_act - Time = {i}, trial {trial_number}")
        axs[0, 1].set_title(f"Field u_sim - Time = {i}")
        axs[1, 0].set_title(f"Field u_f1 - Time = {i}")
        axs[1, 1].set_title(f"Field u_f2 - Time = {i}")
        axs[2, 0].set_title(f"Field u_wm - Time = {i}")

        plt.pause(0.25)

# ====================================
# -------- Post-Processing -----------
# ====================================

# Smooth and accumulate adaptation memory
h_u_amem = gaussian_filter1d(h_u_amem, sigma=15)
if trial_number > 1:
    h_u_amem += h_u_amem + latest_h_amem

# Save adaptation memory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path_1 = results_dir / f"h_u_amem_{timestamp}.npy"
np.save(file_path_1, h_u_amem)
print(f"Saved h_u_amem to {file_path_1}")

# ====================================
# -------- Visualization -------------
# ====================================

# Plot adaptation memory
plt.figure(figsize=(10, 4))
plt.plot(x, h_u_amem, label='h_u_amem')
if trial_number > 1:
    plt.plot(x, latest_h_amem, label='previous', linestyle='--')
plt.xlabel('x')
plt.ylabel('value')
plt.title(f'Change in h_u_amem, trial {trial_number}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert history to arrays
u_act_history = np.array(u_act_history)
u_sim_history = np.array(u_sim_history)
u_wm_history = np.array(u_wm_history)
u_f1_history = np.array(u_f1_history)
u_f2_history = np.array(u_f2_history)

timesteps = np.arange(len(u_act_history))

# Plot field histories
fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

field_histories = [
    (u_act_history, 'u_act', (-2, 5)),
    (u_sim_history, 'u_sim', (-2, 5)),
    (u_wm_history, 'u_wm', (-2, 25)),
    (u_f1_history, 'u_f1', (-2, 5)),
    (u_f2_history, 'u_f2', (-2, 5)),
]

for ax, (field_hist, name, ylim) in zip(axs, field_histories):
    for pos_idx in range(field_hist.shape[1]):
        ax.plot(timesteps, field_hist[:, pos_idx], label=f'x = {input_positions[pos_idx]}')
    ax.set_ylabel(name)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True)

axs[-1].set_xlabel('Time step')
fig.suptitle(f'Field values at input positions over time, trial {trial_number}')
plt.tight_layout()
plt.show()