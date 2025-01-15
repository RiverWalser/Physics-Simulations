import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

class Electron:
    def __init__(self, x_start, y_start, vx_init=0, vy_init=0):
        self.x = x_start
        self.y = y_start
        self.vx = vx_init
        self.vy = vy_init

    def probability(self, x, y):
        # Define parameters for the Gaussian distribution
        sigma = 0.5  # Standard deviation (in nanometers)
        mu_x = self.x  # Mean of x
        mu_y = self.y  # Mean of y

        # Calculate probability using 2D Gaussian distribution formula
        exponent = -((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
        prob = np.exp(exponent) / (2 * np.pi * sigma**2)

        return prob

    def update_position(self, dt, other_electrons, electron_index):
        # Update position based on velocity and time step
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Calculate acceleration due to quantum mechanical interaction
        ax, ay = self.calculate_acceleration(other_electrons)

        # Update velocity based on acceleration
        self.vx += ax * dt
        self.vy += ay * dt

        print("Electron", electron_index)
        print("Position: ({:.2f}, {:.2f})".format(self.x, self.y))
        print("Velocity: ({:.2f}, {:.2f})".format(self.vx, self.vy))
        print("Acceleration: ({:.2f}, {:.2f})".format(ax, ay))
        print()

    def calculate_acceleration(self, other_electrons):
        # Initialize acceleration components
        ax = 0
        ay = 0

        # Constants
        epsilon0 = 8.854e-12  # Vacuum permittivity
        e = 1.602e-19  # Elementary charge
        m_e = 9.109e-31  # Electron mass
        softening_factor = 1e-10  # Softening factor to avoid singularity

        # Calculate acceleration due to other electrons
        for electron in other_electrons:
            if electron != self:
                dx = electron.x - self.x
                dy = electron.y - self.y
                r = np.sqrt(dx**2 + dy**2) + softening_factor  # Softening factor to avoid singularity
                force_mag = (e**2 / (4 * np.pi * epsilon0)) * np.exp(-r) / (r**2 * m_e)
                ax += force_mag * dx / r
                ay += force_mag * dy / r

        return ax, ay

def plot(electron_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for x, y coordinates
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    def update_plot():
        Z = np.zeros_like(X)
        for electron in electron_list:
            Z += electron.probability(X, Y)
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='RdYlGn')
        ax.set_xlabel('X Position (nm)')
        ax.set_ylabel('Y Position (nm)')
        ax.set_zlabel('Probability')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, Z.max())
        plt.draw()

    # Initial plot
    update_plot()

    # Create slider for time
    ax_time = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_time = Slider(ax_time, 'Time', 0, 10, valinit=0, valstep=0.1)  # Set valstep to 0.1

    # Function to update plot based on slider value
    def update_plot_with_slider(val):
        dt = 0.1
        for i, electron in enumerate(electron_list):
            electron.update_position(dt, electron_list, i)
        update_plot()

    # Update plot when slider value changes
    slider_time.on_changed(update_plot_with_slider)

    plt.show()

# Example usage
if __name__ == "__main__":
    # Create a list of electrons with positions and velocities
    electron_list = [
        Electron(0, 0, vx_init=0.1, vy_init=0.2),
        Electron(2, 2, vx_init=-0.1, vy_init=-0.1),
        Electron(-3, 1, vx_init=0.2, vy_init=-0.1)
    ]

    # Plot the electrons
    plot(electron_list)
