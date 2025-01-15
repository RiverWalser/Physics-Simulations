import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
k_e = 8.9875517873681764e9  # Coulomb's constant in N m^2/C^2
e = 1.602176634e-19  # Elementary charge in C
m_e = 9.10938356e-31  # Electron mass in kg
dt = 1e-16  # Time step in seconds, reduced for better simulation resolution
timesteps = 1000  # Number of timesteps

# Initial positions and velocities of the electrons (in meters)
x1, y1 = 0.0, -1.0e-9  # Electron 1 initial position
x2, y2 = 1.0e-9, 0.0  # Electron 2 initial position, 1 nm apart
vx1, vy1 = 0.0, 0.0  # Electron 1 initial velocity
vx2, vy2 = 0.0, 0.0  # Electron 2 initial velocity

# Arrays to store positions, velocities, and accelerations over time
positions1 = np.zeros((timesteps, 2))
positions2 = np.zeros((timesteps, 2))
velocities1 = np.zeros((timesteps, 2))
velocities2 = np.zeros((timesteps, 2))
accelerations1 = np.zeros((timesteps, 2))
accelerations2 = np.zeros((timesteps, 2))

# Store initial positions
positions1[0] = [x1, y1]
positions2[0] = [x2, y2]

# Function to calculate the force on an electron at (x, y) due to another at (x_other, y_other)
def force_on_electron(x, y, x_other, y_other):
    r = np.sqrt((x - x_other)**2 + (y - y_other)**2)
    if r == 0:
        return np.array([0, 0])  # Avoid division by zero
    force_magnitude = k_e * e**2 / r**2
    print(f'force magnitude: {force_magnitude}')
    force_vector = force_magnitude * np.array([(x - x_other) / r, (y - y_other) / r])
    return force_vector

# Time evolution
for t in range(0, timesteps):
    # Calculate forces
    F12 = force_on_electron(x1, y1, x2, y2)
    F21 = -F12

    # Calculate accelerations
    a1 = F12 / m_e
    a2 = F21 / m_e
    if t == 1:
        a1i = a1
        a2i = a2

    # Update velocities
    vx1 += a1[0] * dt
    vy1 += a1[1] * dt
    vx2 += a2[0] * dt
    vy2 += a2[1] * dt

    # Update positions
    x1 += vx1 * dt
    y1 += vy1 * dt
    x2 += vx2 * dt
    y2 += vy2 * dt

    # Store new positions, velocities, and accelerations
    positions1[t] = [x1, y1]
    positions2[t] = [x2, y2]
    velocities1[t] = [vx1, vy1]
    velocities2[t] = [vx2, vy2]
    accelerations1[t] = a1
    accelerations2[t] = a2

# Create the figure for the animation
fig, ax = plt.subplots(figsize=(15, 7))

# Function to update the plot
def update_plot(t):
    plt.clf()
    
    # Determine the bounds for the grid based on the current positions of the electrons
    x_min = min(positions1[t, 0], positions2[t, 0]) - 1e-9
    x_max = max(positions1[t, 0], positions2[t, 0]) + 1e-9
    y_min = min(positions1[t, 1], positions2[t, 1]) - 1e-9
    y_max = max(positions1[t, 1], positions2[t, 1]) + 1e-9
    
    # Create a grid for the slope field (in meters)
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))

    # Initialize the force vectors
    U1, V1 = np.zeros_like(X), np.zeros_like(Y)
    U2, V2 = np.zeros_like(X), np.zeros_like(Y)

    # Calculate the slope field for electron 1 at time t
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            force1 = force_on_electron(X[i, j], Y[i, j], positions2[t, 0], positions2[t, 1])
            U1[i, j], V1[i, j] = force1 / m_e

    # Calculate the slope field for electron 2 at time t
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            force2 = force_on_electron(X[i, j], Y[i, j], positions1[t, 0], positions1[t, 1])
            U2[i, j], V2[i, j] = force2 / m_e

    # Plot the slope fields
    plt.subplot(1, 2, 1)
    plt.quiver(X, Y, U1, V1, color='blue')
    plt.scatter(positions1[t, 0], positions1[t, 1], color='red', label='Electron 1')
    plt.scatter(positions2[t, 0], positions2[t, 1], color='green', label='Electron 2')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.title(f'Timestep {t}: Electron 1')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.quiver(X, Y, U2, V2, color='blue')
    plt.scatter(positions1[t, 0], positions1[t, 1], color='red', label='Electron 1')
    plt.scatter(positions2[t, 0], positions2[t, 1], color='green', label='Electron 2')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.title(f'Timestep {t}: Electron 2')
    plt.legend()
    plt.grid()

# Function to animate the plot
def animate(t):
    update_plot(t)

# Create animation
ani = FuncAnimation(fig, animate, frames=timesteps, interval=1000, repeat=False)

plt.show()

# Plot positions vs time
fig, ax = plt.subplots(3, 2, figsize=(15, 10))

time = np.arange(timesteps) * dt

# Positions
ax[0, 0].plot(time, positions1[:, 0], label='x1')
ax[0, 0].plot(time, positions1[:, 1], label='y1')
ax[0, 0].set_title('Electron 1 Position vs Time')
ax[0, 0].set_xlabel('Time (s)')
ax[0, 0].set_ylabel('Position (m)')
ax[0, 0].legend()
ax[0, 0].grid()

ax[0, 1].plot(time, positions2[:, 0], label='x2')
ax[0, 1].plot(time, positions2[:, 1], label='y2')
ax[0, 1].set_title('Electron 2 Position vs Time')
ax[0, 1].set_xlabel('Time (s)')
ax[0, 1].set_ylabel('Position (m)')
ax[0, 1].legend()
ax[0, 1].grid()

# Velocities
ax[1, 0].plot(time, velocities1[:, 0], label='vx1')
ax[1, 0].plot(time, velocities1[:, 1], label='vy1')
ax[1, 0].set_title('Electron 1 Velocity vs Time')
ax[1, 0].set_xlabel('Time (s)')
ax[1, 0].set_ylabel('Velocity (m/s)')
ax[1, 0].legend()
ax[1, 0].grid()

ax[1, 1].plot(time, velocities2[:, 0], label='vx2')
ax[1, 1].plot(time, velocities2[:, 1], label='vy2')
ax[1, 1].set_title('Electron 2 Velocity vs Time')
ax[1, 1].set_xlabel('Time (s)')
ax[1, 1].set_ylabel('Velocity (m/s)')
ax[1, 1].legend()
ax[1, 1].grid()

# Magnitudes of accelerations
acceleration_magnitudes1 = np.linalg.norm(accelerations1, axis=1)
acceleration_magnitudes2 = np.linalg.norm(accelerations2, axis=1)

ax[2, 0].plot(time, acceleration_magnitudes1, label='|a1|')
ax[2, 0].set_title('Electron 1 Acceleration Magnitude vs Time')
ax[2, 0].set_xlabel('Time (s)')
ax[2, 0].set_ylabel('Acceleration (m/s²)')
ax[2, 0].legend()
ax[2, 0].grid()

ax[2, 1].plot(time, acceleration_magnitudes2, label='|a2|')
ax[2, 1].set_title('Electron 2 Acceleration Magnitude vs Time')
ax[2, 1].set_xlabel('Time (s)')
ax[2, 1].set_ylabel('Acceleration (m/s²)')
ax[2, 1].legend()
ax[2, 1].grid()

plt.tight_layout()
plt.show()

print(f'''electron 1 delta matrix: [ position: [ {0.0 - x1}, {0.0 - y1}]
                                   velocity:[ {0.0 - vx1}, {0.0 - vy1}]
                                   acceleration: [{a1i - a1}, {a2i - a2}]
    ]''')
