import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, m_e, epsilon_0
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation

# Constants
a0 = 1e-9  # initial separation of electrons in meters (1 nm)
eV_to_J = 1.60218e-19  # conversion factor from eV to Joules

# Simulation parameters
L = 5e-9  # length of the simulation box (5 nm)
N = 300  # number of grid points
dx = L / N  # grid spacing
dt = 1e-18  # time step in seconds
T = 1e-15  # total time for the simulation (1 femtosecond)
num_steps = int(T / dt)  # number of time steps

# Create the grid
x = np.linspace(-L / 2, L / 2, N)

# Initial wave function (Gaussian wave packet)
def initial_wave_packet(x, x0, k0, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)

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
    print(np.shape(T))
    return T

H1 = hamiltonian(N, dx)
H2 = hamiltonian(N, dx)

# Time evolution using Crank-Nicolson method
def time_evolution(psi, H, V, dt):
    A = (np.eye(N) - 1j * dt / (2 * hbar) * (H + V))
    B = (np.eye(N) + 1j * dt / (2 * hbar) * (H + V))
    return np.linalg.solve(A, B.dot(psi))

# Set up the figure and axis
fig, ax = plt.subplots()
line1, = ax.plot(x, np.abs(psi1)**2, label='Electron 1')
line2, = ax.plot(x, np.abs(psi2)**2, label='Electron 2')
ax.set_xlim(-L / 2, L / 2)
ax.set_ylim(0, 1.5 * np.max(np.abs(psi1)**2))
ax.set_xlabel('Position (m)')
ax.set_ylabel('Probability Density')
ax.legend()

# Animation function
def animate(step):
    global psi1, psi2
    # Update the wave functions
    psi1 = time_evolution(psi1, H1, V, dt)
    psi2 = time_evolution(psi2, H2, V, dt)
    
    # Normalize the wave functions
    psi1 = normalize(psi1)
    psi2 = normalize(psi2)
    
    # Update the plot data
    line1.set_ydata(np.abs(psi1)**2)
    line2.set_ydata(np.abs(psi2)**2)
    ax.set_ylim(0, 1.5 * np.max(np.abs(psi1)**2))
    
    ax.set_title(f'Time step: {step}')
    return line1, line2

# Create animation
ani = FuncAnimation(fig, animate, frames=num_steps, interval=1, blit=True)

plt.show()
