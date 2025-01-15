import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

class Electron:
    def __init__(self, x_start, y_start, vx, vy):
        self.x = x_start
        self.y = y_start
        self.vx = vx
        self.vy = vy

    def update_position(self, t):
        # Update position based on velocity and time
        self.x = self.x + self.vx * t
        self.y = self.y + self.vy * t

    def probability(self, x, y):
        # Define parameters for the Gaussian distribution
        sigma = 0.5  # Standard deviation (in nanometers)
        mu_x = self.x  # Mean of x
        mu_y = self.y  # Mean of y

        # Calculate probability using 2D Gaussian distribution formula
        exponent = -((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
        prob = np.exp(exponent) / (2 * np.pi * sigma**2)

        return prob

def plot(electron_list, ax, fig):
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    def update_plot(val):
        t = slider.val
        Z = np.zeros_like(X)

        for electron in electron_list:
            electron.update_position(t)
            Z += electron.probability(X, Y)

        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='RdYlGn')
        ax.set_xlabel('X Position (nm)')
        ax.set_ylabel('Y Position (nm)')
        ax.set_zlabel('Probability')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, Z.max())

        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Time', 0, 10, valinit=0, valstep=0.1)

    slider.on_changed(update_plot)

    update_plot(0)  # Initial plot

    return slider

if __name__ == "__main__":
    # Create a list of electrons with initial positions and velocities
    electron_list = [
        Electron(0, 0, 0.1, 0.1),
        Electron(2, 2, -0.05, 0.05),
        Electron(-3, 1, 0.1, -0.1)
    ]

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    slider = plot(electron_list, ax, fig)

    # Set the update interval for the slider (1/10th of its length every second)
    ani = FuncAnimation(fig, lambda i: slider.set_val(slider.val + 0.1), interval=1000)

    plt.show()
