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
dt = .1    # Time step
total_time = 200  # Total simulation time

# Function to calculate the time-evolved standard deviation
def calculate_sigma_t(sigma, t):
    return sigma * np.sqrt(1 + (1j * hbar * t) / (2 * m_e * sigma**2))

# Function to calculate the Gaussian wave function at time t
def gaussian_wavefunction(x, y, sigma_t):
    norm_factor = 1 / np.sqrt(2 * np.pi * sigma_t**2)
    exponent = -((x - x0)**2 + (y - y0)**2) / (2 * sigma_t**2)
    return norm_factor * np.exp(exponent)

# Create meshgrid for x, y values
x_vals = np.linspace(-xmax, xmax, 500)
y_vals = np.linspace(-ymax, ymax, 500)
X, Y = np.meshgrid(x_vals, y_vals)

# Initialize figure for plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Loop over time steps
for t in np.arange(0, total_time, dt):
    # Calculate the time-evolved standard deviation
    sigma_t = calculate_sigma_t(sigma, t)
    
    # Calculate the Gaussian wave function at time t
    psi = gaussian_wavefunction(X, Y, sigma_t)
    
    # Calculate the probability density (|psi|^2)
    probability_density = np.abs(psi)**2
    
    # Plot the probability density as a 3D surface plot
    ax.clear()
    ax.plot_surface(X, Y, probability_density, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Probability Density')
    ax.set_title(f'Time = {t:.2f}')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_zlim(0, 1)  # Adjust z-limit as needed
    plt.pause(0.001)  # Pause to update plot
    
plt.show()
