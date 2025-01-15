import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

# Constants
k_e = 8.99e9  # Coulomb's constant in N m²/C²
e = 1.602e-19  # Charge of an electron in C
m = 9.11e-31  # Mass of an electron in kg
A = np.sqrt(2 * k_e * e**2 / m)
r_0 = 1e-10  # Initial distance in meters

# Coulomb's Law and Electron Dynamics
# Constants:
# k_e: Coulomb's constant = 8.99 × 10^9 N m²/C²
# e: Charge of an electron = 1.602 × 10^-19 C
# m: Mass of an electron = 9.11 × 10^-31 kg

# Derivation:
# Coulomb's law for the force between two electrons:
# F = k_e * e^2 / r^2

# Newton's second law gives:
# F = m * a
# Thus, acceleration a = F / m

# The differential equation for the position r(t) based on Coulomb's force:
# d²r/dt² = k_e * e^2 / (m * r^2)

# Separating variables and integrating:
# ∫ r² dr = ∫ (k_e * e^2 / m) dt²
# Solving the integral:
# r(t) = sqrt(r_0^2 + (A * t / m)^2)
# where A = sqrt(2 * k_e * e^2 / m)

class Electron:
    def __init__(self, x_start, y_start, vx_init=0, vy_init=0, charge=-1.6e-19, mass=9.10938356e-31):
        self.x = x_start
        self.y = y_start
        self.vx = vx_init
        self.vy = vy_init
        self.charge = charge  # Electron charge in coulombs
        self.mass = mass  # Electron mass in kilograms

    def update_position(self, dt, other_electrons):
        # Update position based on velocity and time step
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Calculate forces from other electrons using Coulomb's Law
        for electron in other_electrons:
            if electron != self:
                dx = electron.x - self.x
                dy = electron.y - self.y
                r_squared = dx**2 + dy**2
                r = np.sqrt(r_squared)
                
                # Ensure minimum distance to prevent division by zero
                if r < 1e-6:
                    r = 1e-6

                force_magnitude = (8.9875e9 * self.charge * electron.charge) / r_squared  # Coulomb's Law
                fx = force_magnitude * dx / r
                fy = force_magnitude * dy / r

                # Update velocities based on the force
                self.vx += fx / self.mass * dt
                self.vy += fy / self.mass * dt

    def probability(self, x, y):
        # Define parameters for the Gaussian distribution
        sigma = 0.5  # Standard deviation (in nanometers)
        mu_x = self.x  # Mean of x
        mu_y = self.y  # Mean of y

        # Calculate probability using 2D Gaussian distribution formula
        exponent = -((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
        prob = np.exp(exponent) / (2 * np.pi * sigma**2)

        return prob

def plot(electron_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for x, y coordinates
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Initialize probability grid
    Z = np.zeros_like(X)
    for electron in electron_list:
        Z += electron.probability(X, Y)

    # Plot the surface with red-green colormap
    surface = ax.plot_surface(X, Y, Z, cmap='RdYlGn')

    ax.set_xlabel('X Position (nm)')
    ax.set_ylabel('Y Position (nm)')
    ax.set_zlabel('Probability')

    # Set limits for better zoom
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, Z.max())

    # Create slider for time
    ax_time = fig.add_axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_time = Slider(ax_time, 'Time', 0, 1, valinit=0, valstep=0.00001)

    # Create zoom button
    axzoom = plt.axes([0.02, 0.02, 0.1, 0.04])
    button_zoom = Button(axzoom, 'Zoom')

    def zoom(event):
        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type('ortho' if ax.get_proj_type() == 'persp' else 'persp')
        redraw_plot()

    button_zoom.on_clicked(zoom)

    last_time = [slider_time.val]  # Store the last time value in a list for mutability

    # Update function for slider
    def update(val):
        time = slider_time.val
        dt = time - last_time[0]
        print(f"Updating positions with dt = {dt}")
        for electron in electron_list:
            electron.update_position(dt, electron_list)  # Pass electron_list as other_electrons
        last_time[0] = time  # Update last_time with the current time
        redraw_plot()

    slider_time.on_changed(update)

    def redraw_plot():
        nonlocal surface
        Z = np.zeros_like(X)
        for electron in electron_list:
            Z += electron.probability(X, Y)
        surface.remove()
        surface = ax.plot_surface(X, Y, Z, cmap='RdYlGn')
        plt.draw()

    plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize electrons with higher initial velocities
    electron1 = Electron(0, 0, vx_init=0, vy_init=0)
    electron2 = Electron(2, 2, vx_init=0, vy_init=0)
    electron3 = Electron(-3, 1, vx_init=0, vy_init=0)

    electron_list = [electron1, electron2, electron3]

    plot(electron_list)
