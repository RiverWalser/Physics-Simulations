import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, m_e, epsilon_0
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation

# Constants
a0 = 1e-9  # initial separation of electrons in meters (1 nm)
eV_to_J = 1.60218e-19  # conversion factor from eV to Joules

# Simulation parameters
L = 10e-9  # length of the simulation box (5 nm)
N = 300  # number of grid points
dx = L / N  # grid spacing
dt = 1e-18  # time step in seconds
T = 9e-17  # total time for the simulation (1 femtosecond)
num_steps = int(T / dt)  # number of time steps

# Create the grid
x = np.linspace(-L / 2, L / 2, N)

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, k0, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

# Parameters for the initial wave packet
sigma = 1e-10  # width of the wave packet (0.1 nm)
k0 = 5e10  # initial wave vector

# Initial wave functions of the two electrons
psi1 = initial_wave_packet(x, -a0 / 2, k0, sigma)
psi2 = initial_wave_packet(x, a0 / 2, -k0, sigma)

# Normalization of wave function
def normalize(psi):
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

psi1 = normalize(psi1)
psi2 = normalize(psi2)

# Potential energy function for Coulomb repulsion
def potential_energy(x1, x2):
    r = np.abs(x1 - x2)
    r += 1e-12  # small constant to avoid division by zero
    return e**2 / (4 * np.pi * epsilon_0 * r)

# Create the potential energy matrix
V = np.zeros((N, N), dtype=complex)
for i in range(N):
    for j in range(N):
        V[i, j] = potential_energy(x[i], x[j])

# Hamiltonian for a single particle
def hamiltonian(N, dx):
    # Kinetic energy operator
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray() * (-hbar**2 / (2 * m_e * dx**2))
    return T

H1 = hamiltonian(N, dx)
H2 = hamiltonian(N, dx)

# Time evolution using Crank-Nicolson method
def time_evolution(psi, H, V, dt):
    A = (np.eye(N) - 1j * dt / (2 * hbar) * (H + V))
    B = (np.eye(N) + 1j * dt / (2 * hbar) * (H + V))
    return np.linalg.solve(A, B.dot(psi))

# Calculate the expectation value of position
def expectation_value_position(psi, x):
    return np.real(np.sum(psi.conj() * x * psi) * dx)

# Calculate the expectation value of momentum
def expectation_value_momentum(psi, x):
    dp = hbar / dx
    k = np.fft.fftfreq(len(x), d=dx)
    psi_k = np.fft.fft(psi)
    return np.real(np.sum(psi_k.conj() * k * psi_k) * dp)

# Calculate the expectation value of acceleration
def expectation_value_acceleration(psi, x, V):
    return np.real(np.sum(psi.conj() * (-1j * hbar / m_e * (H1 + V)) * psi) * dx)

# Arrays to store position, velocity, and acceleration over time
times = np.linspace(0, T, num_steps)
position1 = np.zeros(num_steps)
position2 = np.zeros(num_steps)
velocity1 = np.zeros(num_steps)
velocity2 = np.zeros(num_steps)
acceleration1 = np.zeros(num_steps)
acceleration2 = np.zeros(num_steps)

# Set up the figure and axes for probability density
fig1, axs1 = plt.subplots()
line1, = axs1.plot(x, np.abs(psi1)**2, label='Electron 1')
line2, = axs1.plot(x, np.abs(psi2)**2, label='Electron 2')
axs1.set_xlim(-L / 2, L / 2)
axs1.set_ylim(0, 1.5 * np.max(np.abs(psi1)**2))
axs1.set_xlabel('Position (m)')
axs1.set_ylabel('Probability Density')
axs1.legend()

# Animation function
def animate(step):
    print(step)
    global psi1, psi2
    
    # Update the wave functions
    psi1 = time_evolution(psi1, H1, V, dt)
    psi2 = time_evolution(psi2, H2, V, dt)
    
    # Normalize the wave functions
    psi1 = normalize(psi1)
    psi2 = normalize(psi2)
    
    # Update the probability density plot
    line1.set_ydata(np.abs(psi1)**2)
    line2.set_ydata(np.abs(psi2)**2)
    axs1.set_title(f'Time step: {step}')
    
    # Update the expectation values
    position1[step] = expectation_value_position(psi1, x)
    position2[step] = expectation_value_position(psi2, x)
    velocity1[step] = expectation_value_momentum(psi1, x) / m_e
    velocity2[step] = expectation_value_momentum(psi2, x) / m_e
    acceleration1[step] = expectation_value_acceleration(psi1, x, V)
    acceleration2[step] = expectation_value_acceleration(psi2, x, V)
    
    return line1, line2

# Create animation
ani = FuncAnimation(fig1, animate, frames=num_steps, interval=1, blit=True)

plt.show()

# Plot final results after the animation
fig2, axs2 = plt.subplots(3, 1, figsize=(10, 12))

# Position plot
axs2[0].set_xlim(0, T)
axs2[0].set_ylim(-L / 2, L / 2)
axs2[0].set_xlabel('Time (s)')
axs2[0].set_ylabel('Position (m)')
axs2[0].plot(times, position1, label='Electron 1 Position')
axs2[0].plot(times, position2, label='Electron 2 Position')
axs2[0].legend()

# Velocity plot
axs2[1].set_xlim(0, T)
axs2[1].set_ylim(-1e6, 1e6)  # Adjust this range based on expected velocities
axs2[1].set_xlabel('Time (s)')
axs2[1].set_ylabel('Velocity (m/s)')
axs2[1].plot(times, velocity1, label='Electron 1 Velocity')
axs2[1].plot(times, velocity2, label='Electron 2 Velocity')
axs2[1].legend()

# Acceleration plot
axs2[2].set_xlim(0, T)
axs2[2].set_ylim(-1e16, 1e16)  # Adjust this range based on expected accelerations
axs2[2].set_xlabel('Time (s)')
axs2[2].set_ylabel('Acceleration (m/s²)')
axs2[2].plot(times, acceleration1, label='Electron 1 Acceleration')
axs2[2].plot(times, acceleration2, label='Electron 2 Acceleration')
axs2[2].legend()

plt.tight_layout()
plt.show()
