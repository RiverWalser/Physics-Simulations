import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox
from matplotlib.animation import FuncAnimation

class Electron:
    def __init__(self, x_start, y_start, vx_init, vy_init):
        self.x = x_start
        self.y = y_start
        self.vx = vx_init
        self.vy = vy_init
        self.prev_x = x_start
        self.prev_y = y_start

    def probability(self, x, y):
        sigma = 0.5  # Standard deviation (in nanometers)
        mu_x = self.x  # Mean of x
        mu_y = self.y  # Mean of y

        exponent = -((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
        prob = np.exp(exponent) / (2 * np.pi * sigma**2)

        return prob

    def update_position(self, dt, other_electrons, electron_index, epsilon0, e):
        self.prev_x = self.x
        self.prev_y = self.y

        ax, ay = self.calculate_acceleration(other_electrons, epsilon0, e)

        self.vx += ax * dt
        self.vy += ay * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        print("Electron", electron_index)
        print("Position: ({:.2f}, {:.2f})".format(self.x, self.y))
        print("Velocity: ({:.2f}, {:.2f})".format(self.vx, self.vy))
        print("Acceleration: ({:.2f}, {:.2f})".format(ax, ay))
        print()

    def calculate_acceleration(self, other_electrons, epsilon0, e):
        ax = 0
        ay = 0

        for electron in other_electrons:
            if electron != self:
                dx = electron.x - self.x
                dy = electron.y - self.y
                r = np.sqrt(dx**2 + dy**2)
                
                if r < 1e-10:
                    continue
                
                force_mag = (e**2 / (4 * np.pi * epsilon0)) / (r**2)
                ax += force_mag * dx / r
                ay += force_mag * dy / r

        return ax, ay

def plot(electron_list):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    dt = 1e-3  # Initial time step
    epsilon0 = 8.854e-21  # Initial vacuum permittivity
    e = 1.602e-10  # Initial elementary charge

    def update_plot(frame):
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

        for i, electron in enumerate(electron_list):
            electron.update_position(dt, electron_list, i, epsilon0, e)

    ani = FuncAnimation(fig, update_plot, frames=100, interval=1000)

    ax_time = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_time = Slider(ax_time, 'Time', 0, 10, valinit=0, valstep=0.1)

    def update_plot_with_slider(val):
        update_plot(val)

    slider_time.on_changed(update_plot_with_slider)

    ax_speed = plt.axes([0.25, 0.95, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_speed = Slider(ax_speed, 'Simulation Speed', 1, 10, valinit=1, valstep=1)

    def advance_slider(val):
        current_val = slider_time.val
        speed = slider_speed.val
        interval = 1000 / speed
        if current_val < slider_time.valmax:
            slider_time.set_val(current_val + slider_time.valstep)
        else:
            slider_time.set_val(slider_time.valmin)
        ani.event_source.interval = interval

    ani_slider = FuncAnimation(fig, advance_slider, frames=np.arange(0, 10, 0.1), interval=1000)

    ax_perm = plt.axes([0.1, 0.15, 0.1, 0.05])
    text_perm = TextBox(ax_perm, 'Vacuum Permittivity', initial=str(epsilon0))

    ax_eps = plt.axes([0.1, 0.25, 0.1, 0.05])
    text_eps = TextBox(ax_eps, 'Epsilon', initial=str(e))

    def submit_perm(text):
        nonlocal epsilon0
        try:
            epsilon0 = float(text)
        except ValueError:
            pass  # Handle the case where the input is not a valid float

    def submit_eps(text):
        nonlocal e
        try:
            e = float(text)
        except ValueError:
            pass  # Handle the case where the input is not a valid float

    text_perm.on_submit(submit_perm)
    text_eps.on_submit(submit_eps)

    plt.show()

electron_list = [
    Electron(0, 0, vx_init=0, vy_init=0),
    Electron(2, 2, vx_init=0, vy_init=0),
    Electron(-3, 1, vx_init=0, vy_init=0)
]

for i, electron in enumerate(electron_list):
    print("Electron", i)
    print("Initial Velocity: ({:.2f}, {:.2f}) m/s".format(electron.vx, electron.vy))

plot(electron_list)
