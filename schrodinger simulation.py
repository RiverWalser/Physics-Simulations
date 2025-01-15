import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
hbar = 1.0  # Reduced Planck's constant (set to 1 for simplicity)
m_e = 1.0   # Electron mass (set to 1 for simplicity)

# Initial parameters
sigma = 1.0     # Initial standard deviation
x0 = 0.0        # Initial x position of the wave packet
y0 = 0.0        # Initial y position of the wave packet
xmax = 10.0     # Maximum x value for plotting
ymax = 10.0     # Maximum y value for plotting
dt = 1        # Time step
total_time = 200  # Total simulation time
y_shift = 5.0   # Shift for the Coulomb potential

# Function to calculate the time-evolved standard deviation
def calculate_sigma_t(sigma, t):
    return sigma * np.sqrt(1 + (1j * hbar * t) / (2 * m_e * sigma**2))

# Function to calculate the Gaussian wave function at time t
def gaussian_wavefunction(x, y, sigma_t):
    norm_factor = 1 / np.sqrt(2 * np.pi * sigma_t**2)
    exponent = -((x - x0)**2 + (y - y0)**2) / (2 * sigma_t**2)
    return norm_factor * np.exp(exponent)

# Function to calculate the Coulomb potential
def coulomb_potential(x, y, y_shift):
    return 1 / np.sqrt(x**2 + (y - y_shift)**2)

# Function to calculate the force due to Coulomb potential
def coulomb_force(x, y, y_shift):
    r_squared = x**2 + (y - y_shift)**2
    r = np.sqrt(r_squared)
    F = -1 / r_squared * np.array([x, y - y_shift])
    Fx, Fy = F[0], F[1]
    F_magnitude = np.sqrt(Fx**2 + Fy**2)
    return F_magnitude

# Create meshgrid for x, y values
x_vals = np.linspace(-xmax, xmax, 500)
y_vals = np.linspace(-ymax, ymax, 500)
X, Y = np.meshgrid(x_vals, y_vals)

# Initialize figure for plotting
fig = plt.figure(figsize=(20, 8))

# Initialize wave function
sigma_t0 = calculate_sigma_t(sigma, 0)
psi = gaussian_wavefunction(X, Y, sigma_t0)

# Calculate initial probability density and set fixed z-limit
initial_probability_density = np.abs(psi)**2
zlim = (0, np.max(initial_probability_density))

# Time evolution loop
for t in np.arange(0, total_time, dt):
    # Calculate the time-evolved standard deviation
    sigma_t = calculate_sigma_t(sigma, t)
    
    # Calculate the potential at time t
    V = coulomb_potential(X, Y, y_shift)
    
    # Time evolution of the wave function using the split-operator method
    # Apply potential part of the evolution operator
    psi = psi * np.exp(-1j * V * dt / hbar)
    
    # Apply kinetic part of the evolution operator (Fourier space)
    psi_ft = np.fft.fft2(psi)
    kx = np.fft.fftfreq(psi.shape[0], d=(x_vals[1] - x_vals[0]))
    ky = np.fft.fftfreq(psi.shape[1], d=(y_vals[1] - y_vals[0]))
    KX, KY = np.meshgrid(kx, ky)
    kinetic_term = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m_e))
    psi_ft = psi_ft * kinetic_term
    psi = np.fft.ifft2(psi_ft)
    
    # Calculate the probability density (|psi|^2)
    probability_density = np.abs(psi)**2
    
    # Plot the probability density as a 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.clear()
    ax1.plot_surface(X, Y, probability_density, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Probability Density')
    ax1.set_title(f'Time = {t:.2f}')
    ax1.set_xlim(-xmax, xmax)
    ax1.set_ylim(-ymax, ymax)
    ax1.set_zlim(zlim)  # Fixed z-limit
    
    # Plot the force as a 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.clear()
    force = coulomb_force(X, Y, y_shift)
    ax2.plot_surface(X, Y, force, cmap='inferno', edgecolor='none')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('Force Magnitude')
    ax2.set_title('Coulomb Force Magnitude')
    ax2.set_xlim(-xmax, xmax)
    ax2.set_ylim(-ymax, ymax)
    ax2.set_zlim(0, np.max(force))  # Adjust z-limit as needed
    
    plt.pause(0.001)  # Pause to update plot
    
plt.show()
