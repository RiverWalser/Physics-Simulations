import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
e = 1.602e-19  # elementary charge in Coulombs
epsilon_0 = 8.854e-12  # permittivity of free space in Farads per meter

# Positions of electrons (example positions)
electron_positions = np.array([
    [2.0, 1.0e-1],
    [-1.0, -1.0e-1],
    [3.0e-1, -1.5],
    [-0.5e-1, 2.0e-1]
])

# Function to calculate electric potential due to 4 electrons
def electric_potential(x, y):
    V = 0.0
    for pos in electron_positions:
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        V += e / (4 * np.pi * epsilon_0 * distance)
    return V

# Function to calculate electric field magnitude due to 4 electrons
def electric_field_magnitude(x, y):
    Ex, Ey = 0.0, 0.0
    for pos in electron_positions:
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        Ex += e * (x - pos[0]) / (4 * np.pi * epsilon_0 * distance**3)
        Ey += e * (y - pos[1]) / (4 * np.pi * epsilon_0 * distance**3)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    return E_mag

# Create a grid of points
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Calculate electric potential and field magnitude at each point in the grid
Z_potential = electric_potential(X, Y)
Z_field = electric_field_magnitude(X, Y)

# Plotting the electric potential
fig_potential = plt.figure(figsize=(10, 8))
ax_potential = fig_potential.add_subplot(111, projection='3d')
ax_potential.plot_surface(X, Y, Z_potential, cmap='viridis')
ax_potential.set_title('Electric Potential due to 4 Electrons')
ax_potential.set_xlabel('X')
ax_potential.set_ylabel('Y')
ax_potential.set_zlabel('Electric Potential (V)')
plt.show()

# Plotting the electric field magnitude
fig_field = plt.figure(figsize=(10, 8))
ax_field = fig_field.add_subplot(111, projection='3d')
ax_field.plot_surface(X, Y, Z_field, cmap='plasma')
ax_field.set_title('Electric Field Magnitude due to 4 Electrons')
ax_field.set_xlabel('X')
ax_field.set_ylabel('Y')
ax_field.set_zlabel('Electric Field Magnitude (V/m)')
plt.show()
