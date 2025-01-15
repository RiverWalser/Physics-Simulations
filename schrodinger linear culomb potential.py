import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh

# Parameters
L = 50.0  # Length of the domain
N = 1000  # Number of spatial points
x = np.linspace(0, L, N)  # Spatial grid
dx = x[1] - x[0]  # Spatial step size
V0 = 10.0  # Maximum potential at x = 0
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Mass of the particle
t_max = 5.0  # Maximum time
dt = 0.01  # Time step
num_steps = int(t_max / dt)

# Define the potential
V = V0 * (1 - x / L)

# Construct the Hamiltonian matrix
H = np.zeros((N, N))

# Kinetic energy term (finite difference method)
for i in range(1, N - 1):
    H[i, i - 1] = H[i, i + 1] = -hbar**2 / (2 * m * dx**2)
    H[i, i] = hbar**2 / (m * dx**2) + V[i]

# Boundary conditions (infinite potential walls)
H[0, 0] = H[N-1, N-1] = 1e10

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(H)

# Initial wavefunction (a Gaussian packet)
sigma = 0.1  # Width of the packet
x0 = 2.0  # Initial position
k0 = 5.0  # Initial momentum
psi_0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi_0 /= np.linalg.norm(psi_0)  # Normalize the wavefunction

# Expansion coefficients in terms of the eigenfunctions
c_n = np.dot(eigenvectors.T.conj(), psi_0)

# Time evolution
psi_t = np.zeros((num_steps, N), dtype=np.complex128)
psi_t[0, :] = psi_0

for t in range(1, num_steps):
    psi_t[t, :] = np.dot(eigenvectors, c_n * np.exp(-1j * eigenvalues * t * dt))

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, 1.2 * np.max(np.abs(psi_t)**2))
ax.set_xlabel('x')
ax.set_ylabel('|psi(x,t)|^2')
line, = ax.plot([], [], lw=2, label='|psi(x,t)|^2')
potential_line, = ax.plot(x, V / V0 * np.max(np.abs(psi_t)**2), 'r--', lw=1, label='Potential V(x)')

def init():
    line.set_data([], [])
    potential_line.set_data(x, V / V0 * np.max(np.abs(psi_t)**2))
    return line, potential_line

def update(frame):
    line.set_data(x, np.abs(psi_t[frame])**2)
    return line, potential_line

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=20)

plt.legend()
plt.title('Time Evolution of Wavefunction in Linearly Decreasing Potential')
plt.show()
