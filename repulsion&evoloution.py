
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
        self.sigma = 0.1  # Standard deviation (in nanometers), adjusted for actual electron size
        self.psi = self.initialize_wavefunction()

    def initialize_wavefunction(self):
        x_vals = np.linspace(-1, 1, 100, dtype=np.complex128)
        y_vals = np.linspace(-1, 1, 100, dtype=np.complex128)
        X, Y = np.meshgrid(x_vals, y_vals)
        mu_x = np.complex128(self.x)
        mu_y = np.complex128(self.y)
        exponent = -((X - mu_x)**2 + (Y - mu_y)**2) / (2 * self.sigma**2)
        psi = np.exp(exponent) / (2 * np.pi * self.sigma**2)
        return psi

    def probability(self):
        return np.abs(self.psi)**2

    def normalize_wavefunction(self):
        norm = np.sum(np.abs(self.psi)**2)
        if norm > 0:
            self.psi /= np.sqrt(norm)

    def update_position(self, dt, other_electrons, epsilon0, e):
        ax, ay = self.calculate_acceleration(other_electrons, epsilon0, e)

        self.vx += ax * dt
        self.vy += ay * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

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
                ax -= force_mag * dx / r
                ay -= force_mag * dy / r

        return ax, ay

    def update_wavefunction(self, dt, other_electrons, epsilon0, e):
        x_vals = np.linspace(-1, 1, 100, dtype=np.complex128)
        y_vals = np.linspace(-1, 1, 100, dtype=np.complex128)
        X, Y = np.meshgrid(x_vals, y_vals)
        potential = np.zeros_like(X, dtype=np.complex128)

        for electron in other_electrons:
            if electron != self:
                dx = X - np.complex128(electron.x)
                dy = Y - np.complex128(electron.y)
                r = np.sqrt(dx**2 + dy**2)
                r[r < 1e-10] = 1e-10  # Avoid division by zero
                potential += (e**2 / (4 * np.pi * epsilon0)) / r

        hbar = np.complex128(1.0545718e-34)  # Planck constant
        m = np.complex128(9.10938356e-31)  # Electron mass

        laplacian = (np.roll(self.psi, 1, axis=0) + np.roll(self.psi, -1, axis=0) +
                     np.roll(self.psi, 1, axis=1) + np.roll(self.psi, -1, axis=1) -
                     4 * self.psi) / (x_vals[1] - x_vals[0])**2

        self.psi += -1j * dt / hbar * (-hbar**2 / (2 * m) * laplacian + potential * self.psi)
        self.normalize_wavefunction()

def plot(electron_list):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = np.linspace(-1, 1, 100, dtype=np.complex128)
    y_vals = np.linspace(-1, 1, 100, dtype=np.complex128)
    X, Y = np.meshgrid(x_vals, y_vals)

    dt = 1e-16  # Adjusted time step to see more noticeable changes
    epsilon0 = 8.854e-12  # Vacuum permittivity
    e = 1.6e-19  # Elementary charge (adjusted for more noticeable repulsion)

    def update_plot(frame):
        Z = np.zeros_like(X, dtype=np.float64)  # Changed to float64 to avoid complex dtype issues
        for electron in electron_list:
            Z += electron.probability()

        ax.clear()
        ax.plot_surface(X.real, Y.real, Z, cmap='RdYlGn')  # Plot only the real part
        ax.set_xlabel('X Position (nm)')
        ax.set_ylabel('Y Position (nm)')
        ax.set_zlabel('Probability')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, Z.max())

        for i, electron in enumerate(electron_list):
            electron.update_position(dt, electron_list, epsilon0, e)
            electron.update_wavefunction(dt, electron_list, epsilon0, e)

    def advance_slider(val):
        current_val = slider_time.val
        speed = slider_speed.val
        interval = 1000 / speed
        if current_val < slider_time.valmax:
            slider_time.set_val(current_val + slider_time.valstep)
        else:
            slider_time.set_val(slider_time.valmin)
        ani.event_source.interval = interval

    ani = FuncAnimation(fig, update_plot, frames=np.arange(0, 10, 0.1), interval=1000)

    ax_speed = plt.axes([0.25, 0.95, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_speed = Slider(ax_speed, 'Simulation Speed', 1, 10, valinit=1, valstep=1)

    ax_time = plt.axes([0.25, 0.90, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_time = Slider(ax_time, 'Time', 0, 10, valinit=0, valstep=0.1)

    ax_perm = plt.axes([0.1, 0.15, 0.1, 0.05])
    text_perm = TextBox(ax_perm, 'Vacuum Permittivity', initial=str(epsilon0))

    ax_eps = plt.axes([0.1, 0.25, 0.1, 0.05])
    text_eps = TextBox(ax_eps, 'Epsilon', initial=str(e))

    def update_perm(text):
        global epsilon0
        try:
            epsilon0 = float(text)
            update_plot(0)
        except ValueError:
            pass

    def update_eps(text):
        global e
        try:
            e = float(text)
            update_plot(0)
        except ValueError:
            pass

    text_perm.on_submit(update_perm)
    text_eps.on_submit(update_eps)
    ani.event_source.start()
    plt.show()

electron_list = [
    Electron(0, 0, 5.0, 0.0),
    Electron(2, 2, 0.0, 0.0),
    Electron(-3, 1, 0.0, 0.0)
]

for i, electron in enumerate(electron_list):
    print(f"Electron {i}")
    print(f"Initial Velocity: ({electron.vx:.2f}, {electron.vy:.2f}) m/s")

plot(electron_list)
