# pylint: disable=C0200
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from utils import *


# ====================================
# --------- Parameters ---------------
# ====================================


plot_fields = True
# Predefine to avoid linter warnings
fig = axs = line1_field = line1_input = line2_field = line2_input = line1_ud = None

kernel_pars = [1, 0.7, 0.9]

x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05
theta = 1

tau_h = 20
h_0 = 0

h_0_d = 0
tau_h_d = 20
# theta_d = 1.5

input_flag = True
input_shape = [3, 1.5]   # same for both
input_duration = [1, 1, 1, 1, 1]  # same for both

# Positions for input set 1
input_position_1 = [-60, -30, 0, 30, 60]
input_onset_time_1 = [9, 20, 28, 35, 40]

# Positions for input set 2
input_position_2 = input_position_1  # [-50, -30, 10, 35, 65]

# Pack parameters for each input set
input_pars_1 = [input_shape, input_position_1,
                input_onset_time_1, input_duration]


plot_every = 5    # update plot every x time steps
plot_delay = 0.05   # delay (in seconds) before each plot update


# ====================================
# --------- Initialization -----------
# ====================================

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

input_indices_1 = [np.argmin(np.abs(x - pos)) for pos in input_position_1]

inputs_1 = get_inputs(x, t, dt, input_pars_1, input_flag)

u_field_1 = h_0 * np.ones_like(x)

h_u_1 = h_0 * np.ones_like(x)

u_d = h_0_d * np.ones_like(x)
h_u_d = h_0_d * np.ones_like(x)

w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

history_u_1 = np.zeros((len(t), len(x)))
# List to store values at each time step
history_u_d = np.zeros((len(t), len(x)))

# ====================================
# --------- Plot setup ---------------
# ====================================

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)  # rows, col

    # First subplot: field 1
    line1_field, = axs[0].plot(
        x, u_field_1, label='Field activity u_sm(x)')
    line1_input, = axs[0].plot(x, inputs_1[0, :], label='Input 1')

    axs[0].set_ylim(-2, 10)
    axs[0].set_ylabel("Activity")
    axs[0].legend()
    axs[0].set_title("Sequence Memory Field - Time = 0")

    # Second subplot: field u_d
    line2_field, = axs[1].plot(
        x, u_d, label='Field activity u_d(x)')
    axs[1].set_ylim(-2, 10)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Activity")
    axs[1].legend()
    axs[1].set_title("Task Duration Field - Time = 0")


# ====================================
# --------- Simulation loop ----------
# ====================================

# time_counter = 0.0
u_1_tc = []
u_2_tc = []
u_d_tc = []

for i in range(len(t)):

    # Input at time t=0 for the task duration field
    if 0 <= i < 1/dt:
        input_d = 3.0 * np.exp(-((x - 0) ** 2) / (2 * 1.5 ** 2))
    else:
        input_d = 0.0

    f_1 = np.heaviside(u_field_1 - theta, 1)
    f_hat_1 = np.fft.fft(f_1)
    conv_1 = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat_1 * w_hat)))
    h_u_1 += dt / tau_h * f_1
    u_field_1 += dt * (-u_field_1 + conv_1 + inputs_1[i, :] + h_u_1)
    history_u_1[i, :] = u_field_1

    f_d = np.heaviside(u_d - theta, 1)
    f_hat_d = np.fft.fft(f_d)
    conv_d = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat_d * w_hat)))
    h_u_d += dt / tau_h_d * f_d
    u_d += dt * (-u_d + conv_d + input_d + h_u_d)
    history_u_d[i, :] = u_d

    u_1_values = [u_field_1[idx] for idx in input_indices_1]
    u_d_value = u_d[int(len(x) / 2)]

    u_1_tc.append(u_1_values)
    u_d_tc.append(u_d_value)

    # Update plot every plot_every steps or at last step
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        # Update field 1
        line1_field.set_ydata(u_field_1)
        line1_input.set_ydata(inputs_1[i, :])
        axs[0].set_title(f"Sequence Memory Field - Time = {t[i]:.2f}")

        # Update field 2
        line2_field.set_ydata(u_d)
        axs[1].set_title(f"Task Duration Field - Time = {t[i]:.2f}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(plot_delay)


print(f"Max of u_sm1: {max(u_field_1)}")
print(f"Max of u_d: {max(u_d)}")

# ====================================
# --------- Save Final Fields --------
# ====================================

# Create data folder if it doesn't exist
os.makedirs("/home/wwojtak/dnf_architecture_python/data_basic", exist_ok=True)

# Get current date and time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Define file paths
file_path_1 = f"/home/wwojtak/dnf_architecture_python/data_basic/u_field_1_{timestamp}.npy"
file_path_2 = f"/home/wwojtak/dnf_architecture_python/data_basic/u_d_{timestamp}.npy"

# Save final field states
np.save(file_path_1, u_field_1)
np.save(file_path_2, u_d)

print(f"Saved u_field_1 to {file_path_1}")
print(f"Saved u_d to {file_path_2}")


plt.ioff()
plt.show()


u_f1_history = np.array(u_1_tc)
u_d_history = np.array(u_d_tc)

timesteps = np.arange(len(u_f1_history))

# For u_f1_history and u_f2_history: shape (timesteps, positions)
for i, pos in enumerate(input_position_1):
    crossing_idx = np.argmax(u_f1_history[:, i] >= theta)
    print(f"u_field_1 at x={pos} crosses theta at time {crossing_idx*dt}")

print(" ")

# For u_d_history: shape (timesteps,)
crossing_idx = np.argmax(u_d_history >= theta)
print(f"u_d at x=0 crosses theta at time {crossing_idx*dt}")


fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# Plot u_field_1
for i, pos in enumerate(input_position_1):
    axs[0].plot(timesteps, u_f1_history[:, i], label=f'x = {pos}')
axs[0].axhline(theta, color='r', linestyle='--', label='theta = 1')
axs[0].set_ylabel('u_field_1')
axs[0].set_ylim(-1, 5)
axs[0].legend()
axs[0].grid(True)

# Plot u_d
axs[1].plot(timesteps, u_d_history, label='x = 0')
axs[1].axhline(theta, color='r', linestyle='--', label='theta = 1')
axs[1].set_ylabel('u_d')
axs[1].set_xlabel('Timestep')
axs[1].set_ylim(-1, 3)
axs[1].set_xlim(0, 100)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()