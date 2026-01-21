# pylint: disable=C0200
from scipy.ndimage import gaussian_filter1d  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
# requires ffmpeg installed on your system
from matplotlib.animation import FFMpegWriter
from datetime import datetime
import os
from utils import *
from pathlib import Path
import sys

# Project root = current working directory
PROJECT_ROOT = Path.cwd()

# Results directory
results_dir = PROJECT_ROOT / "results_recall"
results_dir.mkdir(exist_ok=True)

# Load most recent files
# data_dir = Path(__file__).resolve().parents[1] / "data"
# file1, file2 = find_latest_field_files(data_dir)

# u_field_1 = np.load(file1)
# u_field_2 = np.load(file2)

# # Create x-axis (based on the field shape)
# x = np.linspace(-80, 80, u_field_1.shape[0])


# # Plot final time step
# plt.figure(figsize=(10, 5))

# plt.subplot(2, 1, 1)
# plt.plot(x, u_field_1)
# plt.title("u_field_1 (final state)")
# plt.xlabel("x")
# plt.ylabel("activity")

# plt.subplot(2, 1, 2)
# plt.plot(x, u_field_2)
# plt.title("u_field_2 (final state)")
# plt.xlabel("x")
# plt.ylabel("activity")

# plt.tight_layout()
# plt.show()

fig = axs = line1_field = line1_input = line2_field = line2_input = line1_ud = None
line3_field = line4_field = line5_field = None


# beta_adapt = 0.0055
beta_adapt = 0.001

# Default to trial 1 if not provided
trial_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Define input_onset_time_2 values for different trials
input_onset_time_by_trial = {
    1: [8, 19, 28, 38, 48],   # Default
    2: [8, 19, 28, 38, 48],  # Example for trial 2
    3: [8, 19, 28, 38, 48],   # You can add more if needed
}

# Use the provided value if available, else fallback to default
input_onset_time_2 = input_onset_time_by_trial.get(
    trial_number, input_onset_time_by_trial[1])

plot_fields = False

plot_every = 5    # update plot every x time steps
plot_delay = 0.05   # delay (in seconds) before each plot update

# ====================================
# --------- Parameters ---------------
# ====================================

kernel_pars_act = [1.5, 0.8, 0.1]  # ok ADDED INHIBITION 0.5
kernel_pars_sim = [1.7, 0.8, 0.7]  # ok
kernel_pars_wm = [1.75, 0.5, 0.8]  # ok
# kernel_pars_wm = [2.0, 0.5, 0.8]  # ok
kernel_pars_f = [1.5, 0.8, 0.0]  # same as in act
kernel_pars_error = [1.5, 0.8, 0.0]  # same as in act

kernel_pars_inh = [3, 1.5, 0.0]

x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05
# theta = 1

# x = np.linspace(-x_lim, x_lim, 200)
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

# # Plot kernels
# plt.figure(figsize=(10, 4))
# plt.plot(x, kernel_act, label='kernel_act: a=1.5, s=0.8, w=0.0')
# plt.plot(x, kernel_sim, label='kernel_sim: a=1.7, s=0.8, w=0.7', linestyle='--')
# plt.title('Gaussian Kernels with Inhibition')
# plt.xlabel('x')
# plt.ylabel('Kernel value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# folder = "/home/wwojtak/dnf_architecture_python/data_basic"
folder = PROJECT_ROOT / "results_learning"

file1, ts1 = find_latest_file_with_prefix(folder, "u_field_1_")
# file2, ts2 = find_latest_file_with_prefix(folder, "u_field_2_")
file3, ts3 = find_latest_file_with_prefix(folder, "u_d_")

# Optional: check if all timestamps match
if not (ts1 == ts3):
    raise ValueError("Timestamps do not match across all files.")

# Load data
u_field_1 = np.load(file1)
# u_field_2 = np.load(file2)
u_d = np.load(file3)

f_test = np.heaviside(u_field_1 - 1.5, 1)

# plt.figure(figsize=(10, 4))
# plt.plot(x, u_field_1, label='u 1')
# plt.plot(x, u_d, label='u d', linestyle='--')
# plt.xlabel('x')
# plt.ylabel(' value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ====================================
# --------- Inputs -------------------
# ====================================


input_flag = True
input_shape = [3, 1.5]   # same for both
input_duration = [5, 5, 5, 5, 5]  # same for both

# Positions for input set 1
input_position_1 = [-60, -30, 0, 30, 60]
input_onset_time_1 = [3, 8, 12, 16, 20]

# Positions for input set 2
input_position_2 = input_position_1  # [-50, -30, 10, 35, 65]
# input_onset_time_2 = [8, 12, 16, 26, 30]

# Pack parameters for each input set
input_pars_1 = [input_shape, input_position_1,
                input_onset_time_1, input_duration]
input_pars_2 = [input_shape, input_position_2,
                input_onset_time_2, input_duration]


inputs_1 = get_inputs(x, t, dt, input_pars_1, input_flag)
inputs_2 = get_inputs(x, t, dt, input_pars_2, input_flag)

input_agent_robot_feedback = np.zeros((len(t), len(x)))


# ====================================
# --------- Initialization -----------
# ====================================

# load fields from files to initialize

try:
    # Flatten and compute initial h from task duration
    u_d = u_d.flatten()
    h_d_initial = max(u_d)

    if trial_number == 1:
        # Use u_field_1 for u_act
        # u_act = u_field_1.flatten() - h_d_initial + 1.5
        u_act = u_field_1 - h_d_initial + 1.5
        input_action_onset = u_field_1.flatten()
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

        # Use u_field_2 for u_sim
        u_sim = u_field_1.flatten() - h_d_initial + 1.5
        input_action_onset_2 = u_field_1.flatten()
        h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

    else:
        # TODO: save and load last memory PLUS h_amem
        # now it's loading original memory so the timng is wrong
        # data_dir = "/home/wwojtak/dnf_architecture_python/data_basic"
        data_dir = folder
        print(f"Loading h_amem from {data_dir}")

        latest_h_amem_file, _ = find_latest_file_with_prefix(
            data_dir, "h_u_amem_")
        latest_h_amem = np.load(latest_h_amem_file, allow_pickle=True)

        # # Load saved current memory instead of u_field_1
        # print(f"Loading current memory from {data_dir}")
        # latest_memory_file, _ = find_latest_file_with_prefix(
        #     data_dir, "u_memory_")
        # saved_memory = np.load(latest_memory_file, allow_pickle=True)

        u_act = u_field_1.flatten() - h_d_initial + 1.5 - latest_h_amem
        input_action_onset = u_field_1.flatten() - latest_h_amem
        # u_act = saved_memory.flatten() - h_d_initial + 1.5 - latest_h_amem
        # input_action_onset = saved_memory.flatten() - latest_h_amem
        h_u_act = -h_d_initial * np.ones(np.shape(x)) + 1.5

        # Use u_field_2 for u_sim (CHECK!!!!)
        u_sim = u_field_1.flatten() - h_d_initial + 1.5
        input_action_onset_2 = u_field_1.flatten()
        h_u_sim = -h_d_initial * np.ones(np.shape(x)) + 1.5

except FileNotFoundError:
    print("No previous sequence memory found, initializing with default values.")
    u_act = np.zeros(np.shape(x))
    h_u_act = -3.2 * np.ones(np.shape(x))


# initialize all fields

# Working memory parameters
h_0_wm = -1.0
theta_wm = 0.8

# kernel_pars_wm = (1.75, 0.5, 0.8)


# kernel_sim = kernel_gauss(x, *kernel_pars_sim)

u_wm = h_0_wm * np.ones(np.shape(x))
h_u_wm = h_0_wm * np.ones(np.shape(x))

# Action onset parameters
tau_h_act = 20
theta_act = 1.5

tau_h_sim = 10
theta_sim = 1.5

theta_error = 1.5

# kernel_pars_act = (1.5, 0.8, 0.0)
# w_hat_act = np.fft.fft(kernel_gauss(*kernel_pars_act))

# kernel_pars_sim = (1.7, 0.8, 0.7)
# w_hat_sim = np.fft.fft(kernel_gauss(*kernel_pars_sim))


# Compute kernels
kernel_act = kernel_gauss(x, *kernel_pars_act)
kernel_sim = kernel_gauss(x, *kernel_pars_sim)
kernel_wm = kernel_osc(x, *kernel_pars_wm)

kernel_inh = kernel_gauss(x, *kernel_pars_inh)

# Compute FFTs
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
u_act_history = []  # Lists to store values at each time step
u_sim_history = []
u_wm_history = []
u_f1_history = []
u_f2_history = []
u_error_history = []

# Adaptation memory field
h_u_amem = np.zeros(np.shape(x))


if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(14, 14), sharex=True)

    # --- Top-left: u_act ---
    line1_field, = axs[0, 0].plot(x, u_act, label='u_act(x)')
    axs[0, 0].set_ylim(-5, 5)
    axs[0, 0].set_ylabel("Activity")
    axs[0, 0].legend()
    axs[0, 0].set_title("Field u_act - Time = 0")

    # --- Top-right: u_sim ---
    line2_field, = axs[0, 1].plot(x, u_sim, label='u_sim(x)')
    axs[0, 1].set_ylim(-5, 5)
    axs[0, 1].legend()
    axs[0, 1].set_title("Field u_sim - Time = 0")

    # --- Middle-left: u_f1 ---
    line3_field, = axs[1, 0].plot(x, u_f1, label='u_f1(x)')
    axs[1, 0].set_ylim(-5, 5)
    axs[1, 0].set_ylabel("Activity")
    axs[1, 0].legend()
    axs[1, 0].set_title("Field u_f1 - Time = 0")

    # --- Middle-right: u_f2 ---
    line4_field, = axs[1, 1].plot(x, u_f2, label='u_f2(x)')
    axs[1, 1].set_ylim(-5, 5)
    axs[1, 1].legend()
    axs[1, 1].set_title("Field u_f2 - Time = 0")

    # --- Bottom-left: u_wm ---
    line5_field, = axs[2, 0].plot(x, u_wm, label='u_wm(x)')
    axs[2, 0].set_ylim(-5, 5)
    axs[2, 0].set_xlabel("x")
    axs[2, 0].set_ylabel("Activity")
    axs[2, 0].legend()
    axs[2, 0].set_title("Field u_wm - Time = 0")

    # --- Bottom-right: placeholder for future or empty ---
    axs[2, 1].axis("off")  # Optional: leave blank for now

    plt.tight_layout()


input_positions = input_position_1
input_indices = [np.argmin(np.abs(x - pos)) for pos in input_positions]
threshold_crossed = {pos: False for pos in input_positions}

active_gaussian_inputs = {}  # position -> (start_i, duration)

gaussian_amplitude = 3
gaussian_width = 1.5
input_duration_timesteps = 10


def gaussian_input(x, center, amplitude, width):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


# MAIN LOOP
for i in range(len(t)):

    input_agent_2 = inputs_2[i, :]
    # input_agent_2 = inputs_2[i, :]
    # --- Generate Gaussian input field ---
    # input_agent_robot_feedback = np.zeros_like(x)

    # todo: inputs
    # n = len(latest_input_slice) // 3

    # # Split the received data back into two matrices
    # input_agent1 = latest_input_slice[:n]
    # input_agent2 = latest_input_slice[n:2*n]

    # input_agent_robot_feedback = latest_input_slice[2*n:]

    f_f1_test = np.heaviside(u_f1 - 2, 1)

    f_f1 = np.heaviside(u_f1 - theta_f, 1)
    f_hat_f1 = np.fft.fft(f_f1)
    conv_f1 = dx * \
        np.fft.ifftshift(np.real(np.fft.ifft(f_hat_f1 * w_hat_f)))

    f_f2_test = np.heaviside(u_f2 - 2, 1)

    f_f2 = np.heaviside(u_f2 - theta_f, 1)
    f_hat_f2 = np.fft.fft(f_f2)
    conv_f2 = dx * \
        np.fft.ifftshift(np.real(np.fft.ifft(f_hat_f2 * w_hat_f)))

    f_act = np.heaviside(u_act - theta_act, 1)
    f_hat_act = np.fft.fft(f_act)
    conv_act = dx * \
        np.fft.ifftshift(np.real(np.fft.ifft(f_hat_act * w_hat_act)))

    f_sim = np.heaviside(u_sim - theta_sim, 1)
    f_hat_sim = np.fft.fft(f_sim)
    conv_sim = dx * \
        np.fft.ifftshift(np.real(np.fft.ifft(f_hat_sim * w_hat_sim)))

    f_wm = np.heaviside(u_wm - theta_wm, 1)
    f_hat_wm = np.fft.fft(f_wm)
    conv_wm = dx * \
        np.fft.ifftshift(np.real(np.fft.ifft(f_hat_wm * w_hat_wm)))

    conv_inh = dx * \
        np.fft.ifftshift(np.real(np.fft.ifft(f_hat_wm * w_hat_inh)))

    f_error = np.heaviside(u_error - theta_error, 1)
    f_hat_error = np.fft.fft(f_error)
    conv_error = dx * \
        np.fft.ifftshift(
            np.real(np.fft.ifft(f_hat_error * w_hat_act)))

    # Update field states
    h_u_act += dt / tau_h_act
    h_u_sim += dt / tau_h_sim

    u_act += dt * (-u_act + conv_act + input_action_onset +
                   h_u_act - 6.0 * f_wm * conv_wm)

    u_sim += dt * (-u_sim + conv_sim + input_action_onset_2 +
                   h_u_sim - 6.0 * f_wm * conv_wm)

    u_wm += (dt/1.25) * (-u_wm + conv_wm +
                         8 * ((f_f1 * u_f1) * (f_f2 * u_f2)) + h_u_wm)

    u_f1 += dt * (-u_f1 + conv_f1 + input_agent_robot_feedback[i, :] +
                  h_f - 1 * f_wm * conv_wm)

    u_f2 += dt * (-u_f2 + conv_f2 + input_agent_2 +
                  h_f - 1 * f_wm * conv_wm)
    # u_f1 += dt * (-u_f1 + conv_f1 + input_agent_robot_feedback[i, :] +
    #               h_f - 1 * f_test * conv_inh)

    # u_f2 += dt * (-u_f2 + conv_f2 + input_agent_2 +
    #               h_f - 1 * f_test * conv_inh)

    u_error += dt * (-u_error + conv_error +
                     h_f - 2 * f_sim * conv_sim)

    h_u_amem += beta_adapt * (1 - (f_f2 * f_f1)) * (f_f1 - f_f2)
    # h_u_amem += beta_adapt * \
    #     (1 - (f_f2_test * f_f1_test)) * (f_f1_test - f_f2_test)
    #  change f ?

    # # List of input positions where we previously applied inputs
    # input_positions = [-40, 0, 40]

    # # Convert `input_positions` to indices in `x`
    # input_indices = [np.argmin(np.abs(x - pos)) for pos in input_positions]

    # Store the values at the specified positions in history arrays
    u_act_values_at_positions = [u_act[idx] for idx in input_indices]
    u_act_history.append(u_act_values_at_positions)

    u_sim_values_at_positions = [u_sim[idx] for idx in input_indices]
    u_sim_history.append(u_sim_values_at_positions)

    u_wm_values_at_positions = [u_wm[idx] for idx in input_indices]
    u_wm_history.append(u_wm_values_at_positions)

    u_f1_values_at_positions = [u_f1[idx] for idx in input_indices]
    u_f1_history.append(u_f1_values_at_positions)

    u_f2_values_at_positions = [u_f2[idx] for idx in input_indices]
    u_f2_history.append(u_f2_values_at_positions)

    # TODO: MAKE INPUT LAST LONGER, UNTIL DELETED
    # --- Detect threshold crossing ---
    for idx, pos in zip(input_indices, input_positions):
        if not threshold_crossed[pos] and u_act[idx] > theta_act:
            print(f"Threshold crossed at position {pos} and time {i*dt}")
            threshold_crossed[pos] = True

            time_on = i + 20   # HOW LONG MAKE THE DELAY?
            time_off = len(t)  # ACTIVE TILL THE END OR UNTIL OVERWRITTEN
            gaussian = gaussian_amplitude * \
                np.exp(-((x - pos) ** 2) / (2 * gaussian_width ** 2))
            input_agent_robot_feedback[time_on:time_off, :] += gaussian
            # IF LAST EVENT:
            # print(
            #     f"Max of input_agent_robot_feedback: {max(input_agent_robot_feedback[i, :])}")

    # Update plot every plot_every steps or at last step
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        line1_field.set_ydata(u_act)
        line2_field.set_ydata(u_sim)
        line3_field.set_ydata(u_f1)
        line4_field.set_ydata(u_f2)
        line5_field.set_ydata(u_wm)

        # Update titles with current time
        axs[0, 0].set_title(f"Field u_act - Time = {i}, trial {trial_number}")
        axs[0, 1].set_title(f"Field u_sim - Time = {i}")
        axs[1, 0].set_title(f"Field u_f1 - Time = {i}")
        axs[1, 1].set_title(f"Field u_f2 - Time = {i}")
        axs[2, 0].set_title(f"Field u_wm - Time = {i}")

        plt.pause(0.25)

        # # Update field 1
        # line1_field.set_ydata(u_act)
        # # line1_input.set_ydata(inputs_1[i, :])
        # # line1_ud.set_ydata(u_d)  # <-- update u_d line
        # axs[0].set_title(f"Field 1 - Time = {t[i]:.2f}")

        # # Update field 2
        # line2_field.set_ydata(u_sim)
        # # line2_input.set_ydata(inputs_2[i, :])
        # axs[1].set_title(f"Field 2 - Time = {t[i]:.2f}")

        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # plt.pause(plot_delay)

        # if save_video:
        #     writer.grab_frame()

h_u_amem = gaussian_filter1d(h_u_amem, sigma=15)

if trial_number > 1:
    h_u_amem += h_u_amem + latest_h_amem

# Get current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define file paths
# file_path_1 = f"/home/wwojtak/dnf_architecture_python/data_basic/h_u_amem_{timestamp}.npy"

file_path_1 = results_dir / f"h_u_amem_{timestamp}.npy"

# Save final h_u_amem
np.save(file_path_1, h_u_amem)
print(f"Saved h_u_amem to {file_path_1}")

# if trial_number == 1:


# else:

#     # Save current memory (learned u_field_1 - latest_h_amem)
#     current_memory = u_field_1.flatten() - latest_h_amem
#     memory_filename = f"u_memory_{timestamp}.npy"
#     memory_path = os.path.join("data", memory_filename)
#     np.save(memory_path, current_memory)
#     print(f"Saved current memory to {memory_path}")

# h_clean = gaussian_filter1d(h_u_amem, sigma=10)


plt.figure(figsize=(10, 4))
plt.plot(x, h_u_amem, label='h_u_amem')
# plt.plot(x, h_clean, label='cleaned')
# plt.plot(x, u_act, label='act', linestyle='--')
if trial_number > 1:
    plt.plot(x, latest_h_amem, label='previous', linestyle='--')
plt.xlabel('x')
plt.ylabel(' value')
plt.title(f'Change in h_u_amem, trial {trial_number}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Convert history lists to NumPy arrays
u_act_history = np.array(u_act_history)
u_sim_history = np.array(u_sim_history)
u_wm_history = np.array(u_wm_history)
u_f1_history = np.array(u_f1_history)
u_f2_history = np.array(u_f2_history)

timesteps = np.arange(len(u_act_history))

# Set up figure with 5 subplots
fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=False)

field_histories = [
    (u_act_history, 'u_act'),
    (u_sim_history, 'u_sim'),
    (u_wm_history, 'u_wm'),
    (u_f1_history, 'u_f1'),
    (u_f2_history, 'u_f2'),
]

for ax, (field_hist, name) in zip(axs, field_histories):
    for pos_idx in range(field_hist.shape[1]):
        ax.plot(timesteps, field_hist[:, pos_idx],
                label=f'x = {input_positions[pos_idx]}')
    ax.set_ylabel(name)
    # Custom y-limits
    if name == 'u_wm':
        ax.set_ylim(-2, 25)
    else:
        ax.set_ylim(-2, 5)
    ax.legend()
    ax.grid(True)

axs[-1].set_xlabel('Time step')
fig.suptitle(
    f'Field values at input positions over time, trial {trial_number}')
plt.tight_layout()
plt.show()