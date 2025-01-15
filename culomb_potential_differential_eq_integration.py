# Derivation:
# Given two electrons with initial positions r1(0), r2(0), initial velocities v1(0), v2(0), and initial accelerations a1(0), a2(0)
# due to their repulsive forces, we want to find the change in position over a time interval Δt assuming acceleration decreases linearly.

# Define:
# r1(0), r2(0) = initial positions of electron 1 and electron 2
# v1(0), v2(0) = initial velocities of electron 1 and electron 2
# a1(0), a2(0) = initial accelerations of electron 1 and electron 2
# Δt = time step

# Assume acceleration decreases linearly:
# a1(t) = a1(0) + (a1(Δt) - a1(0)) / Δt * t
# a2(t) = a2(0) + (a2(Δt) - a2(0)) / Δt * t

# Integrate acceleration to get velocity:
# v1(t) = v1(0) + ∫[0 to t] a1(τ) dτ
# v2(t) = v2(0) + ∫[0 to t] a2(τ) dτ

# Since a1(t) and a2(t) are linear functions:
# v1(t) = v1(0) + a1(0)t + (a1(Δt) - a1(0)) / Δt * t^2 / 2
# v2(t) = v2(0) + a2(0)t + (a2(Δt) - a2(0)) / Δt * t^2 / 2

# Integrate velocity to get position:
# r1(t) = r1(0) + ∫[0 to t] v1(τ) dτ
# r2(t) = r2(0) + ∫[0 to t] v2(τ) dτ

# With v1(t) and v2(t) from above, evaluating the integrals:
# r1(t) = r1(0) + v1(0)t + a1(0)t^2 / 2 + (a1(Δt) - a1(0)) / Δt * t^3 / 6
# r2(t) = r2(0) + v2(0)t + a2(0)t^2 / 2 + (a2(Δt) - a2(0)) / Δt * t^3 / 6

# Calculate position change over time step Δt:
# Δr1 = r1(Δt) - r1(0) = v1(0)Δt + a1(0)Δt^2 / 2 + (a1(Δt) - a1(0)) / Δt * Δt^3 / 6
# Δr2 = r2(Δt) - r2(0) = v2(0)Δt + a2(0)Δt^2 / 2 + (a2(Δt) - a2(0)) / Δt * Δt^3 / 6

# This method provides the change in position for each electron over the time interval Δt while accounting for linearly decreasing acceleration.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
k_e = 8.9875517873681764e9  # Coulomb's constant in N m^2/C^2
e = 1.602176634e-19  # Elementary charge in C
m_e = 9.10938356e-31  # Electron mass in kg
dt = 1e-16  # Time step in seconds, reduced for better simulation resolution
timesteps = 100  # Number of timesteps

# Initial positions and velocities of the electrons (in nm)
x1, y1 = 0.0, -1.0e-9  # Electron 1 initial position
x2, y2 = 1.0e-9, 0.0  # Electron 2 initial position, 1 nm apart
vx1, vy1 = 0.0, 0.0  # Electron 1 initial velocity
vx2, vy2 = 0.0, 0.0  # Electron 2 initial velocity

# Arrays to store positions over time
positions1 = np.zeros((timesteps, 2))
positions2 = np.zeros((timesteps, 2))

# Store initial positions
positions1[0] = [x1, y1]
positions2[0] = [x2, y2]

# Function to calculate the force on an electron at (x, y) due to another at (x_other, y_other)
def force_on_electron(x, y, x_other, y_other):
    r = np.sqrt((x - x_other)**2 + (y - y_other)**2)
    if r == 0:
        return np.array([0, 0])  # Avoid division by zero
    force_magnitude = k_e * e**2 / r**2
    force_vector = force_magnitude * np.array([(x - x_other) / r, (y - y_other) / r])
    return force_vector

# Time evolution
for t in range(1, timesteps):
    # Calculate forces
    F12 = force_on_electron(x1, y1, x2, y2)
    F21 = -F12
    
    # Calculate accelerations
    a1 = F12 / m_e
    a2 = F21 / m_e
    
    if t == 1:
        a1i = a1
        a2i = a2
    
    # Calculate next accelerations assuming linear decrease
    next_F12 = force_on_electron(x1 + vx1 * dt, y1 + vy1 * dt, x2 + vx2 * dt, y2 + vy2 * dt)
    next_F21 = -next_F12
    
    next_a1 = next_F12 / m_e
    next_a2 = next_F21 / m_e
    
    # Calculate position change using the derived method
    delta_x1 = vx1 * dt + a1[0] * dt**2 / 2 + (next_a1[0] - a1[0]) / dt * dt**3 / 6
    delta_y1 = vy1 * dt + a1[1] * dt**2 / 2 + (next_a1[1] - a1[1]) / dt * dt**3 / 6
    
    delta_x2 = vx2 * dt + a2[0] * dt**2 / 2 + (next_a2[0] - a2[0]) / dt * dt**3 / 6
    delta_y2 = vy2 * dt + a2[1] * dt**2 / 2 + (next_a2[1] - a2[1]) / dt * dt**3 / 6
    
    # Update positions
    x1 += delta_x1
    y1 += delta_y1
    x2 += delta_x2
    y2 += delta_y2
    
    # Update velocities
    vx1 += (a1[0] + next_a1[0]) / 2 * dt
    vy1 += (a1[1] + next_a1[1]) / 2 * dt
    vx2 += (a2[0] + next_a2[0]) / 2 * dt
    vy2 += (a2[1] + next_a2[1]) / 2 * dt
    
    # Store new positions
    positions1[t] = [x1, y1]
    positions2[t] = [x2, y2]
    
    print(f'''electron1 information matrix: [(x,y):({x1},{y1}),
                                           (vx,vy):({vx1},{vy1})
                                           (a): {a1}
          )]''')
    print(f'''electron2 information matrix: [(x,y):({x2},{y2}),
                                           (vx,vy):({vx2},{vy2})
                                           (a): {a2}
          )]''')

# Create a grid for the slope field (in nm)
X, Y = np.meshgrid(np.linspace(-2e-9, 2e-9, 20), np.linspace(-2e-9, 2e-9, 20))

# Initialize the force vectors
U1, V1 = np.zeros_like(X), np.zeros_like(Y)
U2, V2 = np.zeros_like(X), np.zeros_like(Y)

# Function to update the plot
def update_plot(t):
    plt.clf()
    
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

# Create the figure
fig, ax = plt.subplots(figsize=(15, 7))

# Function to animate the plot
def animate(t):
    update_plot(t)

# Create animation
ani = FuncAnimation(fig, animate, frames=timesteps, interval=1000, repeat=False)

plt.show()

print(f'''electron 1 delta matrix: [ positon: [ {0.0-x1}, {0.0-y1}]
                                   velocity:[ {0- vx1}, {0-vy1}]
                                   acceleration: [{a1i-a1}, {a2i-a2}]
    ]''')
