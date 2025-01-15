import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, m_e, epsilon_0
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Constants
a0 = 1e-9  # initial separation of electrons in meters (1 nm)
eV_to_J = 1.60218e-19  # conversion factor from eV to Joules

# Simulation parameters
L = 4e-9  # length of the simulation box (5 nm)
N = 300  # number of grid points
dx = L / N  # grid spacing
dt = 1e-18  # time step in seconds
T = 1e-15  # total time for the simulation (1 femtosecond)
num_steps = int(T / dt)  # number of time steps

# Create the grid
x = np.linspace(-L / 2, L / 2, N)
y = np.linspace(-L / 2, L / 2, N)
X, Y = np.meshgrid(x, y)

# Initial wave function (2D Gaussian wave packet)
def initial_wave_packet(x, y, x0, y0, kx, ky, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * np.exp(1j * (kx * x + ky * y))

# Parameters for the initial wave packet
sigma = 1e-10  # width of the wave packet (0.1 nm)
kx = 5e10  # initial wave vector in x direction
ky = 5e10  # initial wave vector in y direction

# Initial wave functions of the two electrons
psi1 = initial_wave_packet(X, Y, -a0 / 2, -a0 / 2, kx, ky, sigma)
psi2 = initial_wave_packet(X, Y, a0 / 2, a0 / 2, -kx, -ky, sigma)

# Normalization of wave function
def normalize(psi):
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx * dx)

psi1 = normalize(psi1)
psi2 = normalize(psi2)

# Potential energy function for Coulomb repulsion
def potential_energy(x1, y1, x2, y2):
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    r += 1e-12  # small constant to avoid division by zero
    return e**2 / (4 * np.pi * epsilon_0 * r)

# Create the potential energy matrix
V = np.zeros((N, N, N, N), dtype=complex)
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                V[i, j, k, l] = potential_energy(x[i], y[j], x[k], y[l])

# Hamiltonian for a single particle in 2D
def hamiltonian_2d(N, dx):
    T = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray() * (-hbar**2 / (2 * m_e * dx**2))
    I = np.eye(N)
    return np.kron(I, T) + np.kron(T, I)

H1 = hamiltonian_2d(N, dx)
H2 = hamiltonian_2d(N, dx)

# Time evolution using Crank-Nicolson method in 2D
def time_evolution(psi, H, V, dt):
    A = (np.eye(N**2) - 1j * dt / (2 * hbar) * (H + V.reshape(N**2, N**2)))
    B = (np.eye(N**2) + 1j * dt / (2 * hbar) * (H + V.reshape(N**2, N**2)))
    return np.linalg.solve(A, B.dot(psi.reshape(N**2))).reshape(N, N)

# Create the directory to save frames
output_dir = 'frames trial 4'
os.makedirs(output_dir, exist_ok=True)

# Set up the figure and axis for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
    ax.clear()
    ax.plot_surface(X, Y, np.abs(psi1)**2, cmap='viridis', alpha=1.0)
    ax.plot_surface(X, Y, np.abs(psi2)**2, cmap='plasma', alpha=1.0)
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-L / 2, L / 2)
    ax.set_zlim(0, np.max(np.abs(psi1)**2 + np.abs(psi2)**2))
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Probability Density')
    ax.set_title(f'Time step: {step}')
    
    # Save the current frame
    filename = os.path.join(output_dir, f'frame_{step:04d}.png')
    plt.savefig(filename)
    
    return ax,

# Create animation
ani = FuncAnimation(fig, animate, frames=num_steps, interval=20, blit=False)

plt.show()

