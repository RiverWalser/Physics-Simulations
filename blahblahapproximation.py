import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
hbar = 1.0
c = 137.036
m = 1.0
e = 10.0
epsilon_0 = 1.0 / (4 * np.pi)

# Simulation parameters
L = 100.0
N = 1000
dx = L / N
x = np.linspace(0, L, N, endpoint=False)
dt = 0.001
Nt = 10000

# Momentum space
p = 2 * np.pi * np.fft.fftfreq(N, dx)

# Energy operator in momentum space
E_p = np.sqrt((p*c)**2 + (m*c**2)**2)

def gaussian_wave_packet(x0, sigma, k0):
    return np.exp(-0.5 * ((x - x0 + L/2) % L - L/2)**2 / sigma**2) * np.exp(1j * k0 * x)

# Initialize wavefunctions
k0 = 0.5
psi1 = np.zeros((2, N), dtype=complex)
psi2 = np.zeros((2, N), dtype=complex)

psi1[0] = gaussian_wave_packet(L/4, 2.0, k0)
psi1[1] = psi1[0] * k0 * c / (E_p[0] + m*c**2)
psi1 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx)

psi2[0] = gaussian_wave_packet(3*L/4, 2.0, -k0)
psi2[1] = psi2[0] * (-k0) * c / (E_p[0] + m*c**2)
psi2 /= np.sqrt(np.sum(np.abs(psi2)**2) * dx)

def coulomb_potential(x1, x2, L, softening=0.1):
    r = np.minimum((x1 - x2) % L, (x2 - x1) % L)
    return e**2 / (4 * np.pi * epsilon_0 * np.sqrt(r**2 + softening**2))

def coulomb_force(x1, x2, L, softening=0.1):
    r = (x1 - x2 + L/2) % L - L/2
    return e**2 * r / (4 * np.pi * epsilon_0 * (r**2 + softening**2)**1.5)

def rk4_step(psi, V, dt):
    k1 = -1j * (np.fft.ifft(E_p * np.fft.fft(psi, axis=1), axis=1) + V * psi) / hbar
    k2 = -1j * (np.fft.ifft(E_p * np.fft.fft(psi + 0.5*dt*k1, axis=1), axis=1) + V * (psi + 0.5*dt*k1)) / hbar
    k3 = -1j * (np.fft.ifft(E_p * np.fft.fft(psi + 0.5*dt*k2, axis=1), axis=1) + V * (psi + 0.5*dt*k2)) / hbar
    k4 = -1j * (np.fft.ifft(E_p * np.fft.fft(psi + dt*k3, axis=1), axis=1) + V * (psi + dt*k3)) / hbar
    return psi + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Classical simulation setup
x1_classical = L/4
x2_classical = 3*L/4
v1_classical = k0 * hbar / m
v2_classical = -k0 * hbar / m


def evolve_wavefunction(psi, V, dt):
    # Kinetic energy evolution
    psi_p = np.fft.fft(psi, axis=1)
    psi_p = np.exp(-1j * E_p * dt / (2 * hbar)) * psi_p
    psi = np.fft.ifft(psi_p, axis=1)
    
    # Potential energy evolution
    psi = np.exp(-1j * V * dt / hbar) * psi
    
    # Kinetic energy evolution
    psi_p = np.fft.fft(psi, axis=1)
    psi_p = np.exp(-1j * E_p * dt / (2 * hbar)) * psi_p
    psi = np.fft.ifft(psi_p, axis=1)
    
    return psi

# Time evolution
psi1_history = []
psi2_history = []
V_history = []
x1_classical_history = []
x2_classical_history = []

for t in range(Nt):
    # Quantum evolution
    rho1 = np.abs(psi1[0])**2 + np.abs(psi1[1])**2
    rho2 = np.abs(psi2[0])**2 + np.abs(psi2[1])**2
    V1 = np.array([np.sum(coulomb_potential(xi, x, L) * rho2) * dx for xi in x])
    V2 = np.array([np.sum(coulomb_potential(xi, x, L) * rho1) * dx for xi in x])
    
    psi1 = evolve_wavefunction(psi1, V1, dt)
    psi2 = evolve_wavefunction(psi2, V2, dt)
    
    psi1 /= np.sqrt(np.sum(np.abs(psi1)**2) * dx)
    psi2 /= np.sqrt(np.sum(np.abs(psi2)**2) * dx)
    
    # Classical evolution
    f = coulomb_force(x1_classical, x2_classical, L)
    v1_classical += f * dt / m
    v2_classical -= f * dt / m
    x1_classical = (x1_classical + v1_classical * dt) % L
    x2_classical = (x2_classical + v2_classical * dt) % L
    
    # Store history
    psi1_history.append(psi1.copy())
    psi2_history.append(psi2.copy())
    V_history.append(V1 + V2)
    x1_classical_history.append(x1_classical)
    x2_classical_history.append(x2_classical)
    
    if t % 1000 == 0:
        x1 = np.sum(x * rho1) * dx
        x2 = np.sum(x * rho2) * dx
        p1 = np.sum(p * np.abs(np.fft.fft(psi1[0]))**2) / N
        p2 = np.sum(p * np.abs(np.fft.fft(psi2[0]))**2) / N
        
        print(f"Time step {t}:")
        print(f"  Quantum - Electron 1: pos={x1:.2f}, p={p1:.2f}")
        print(f"  Quantum - Electron 2: pos={x2:.2f}, p={p2:.2f}")
        print(f"  Classical - Electron 1: pos={x1_classical:.2f}, v={v1_classical:.2f}")
        print(f"  Classical - Electron 2: pos={x2_classical:.2f}, v={v2_classical:.2f}")
        print(f"  Max Coulomb potential: {np.max(V1 + V2):.2f}")
        print(f"  Total energy (quantum): {np.sum((E_p * (np.abs(np.fft.fft(psi1, axis=1))**2 + np.abs(np.fft.fft(psi2, axis=1))**2) + (V1+V2) * (rho1+rho2)) * dx) / N:.2f}")
        print()
# Animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

line1, = ax1.plot([], [], 'r', label='Electron 1 (Quantum)')
line2, = ax1.plot([], [], 'b', label='Electron 2 (Quantum)')
line3, = ax1.plot([], [], 'g', label='Coulomb Potential')
line4, = ax1.plot([], [], 'k--', label='Electron 1 (Classical)')
line5, = ax1.plot([], [], 'm--', label='Electron 2 (Classical)')
ax1.set_ylabel('|ψ|² / V(x)')
ax1.legend()

line6, = ax2.plot([], [], 'r', label='Re(ψ₁)')
line7, = ax2.plot([], [], 'b', label='Im(ψ₁)')
line8, = ax2.plot([], [], 'g', label='Re(ψ₂)')
line9, = ax2.plot([], [], 'm', label='Im(ψ₂)')
ax2.set_xlabel('Position')
ax2.set_ylabel('ψ')
ax2.legend()

def animate(i):
    psi1 = psi1_history[i]
    psi2 = psi2_history[i]
    V = V_history[i]
    
    rho1 = np.abs(psi1[0])**2 + np.abs(psi1[1])**2
    rho2 = np.abs(psi2[0])**2 + np.abs(psi2[1])**2
    
    line1.set_data(x, rho1)
    line2.set_data(x, rho2)
    line3.set_data(x, V / np.max(V) * np.max(rho1))  # Scaling potential for visibility
    line4.set_data([x1_classical_history[i]], [np.max(rho1)])
    line5.set_data([x2_classical_history[i]], [np.max(rho1)])
    
    line6.set_data(x, psi1[0].real)
    line7.set_data(x, psi1[0].imag)
    line8.set_data(x, psi2[0].real)
    line9.set_data(x, psi2[0].imag)
    
    ax1.set_title(f'Time: {i*dt:.2f}')
    return line1, line2, line3, line4, line5, line6, line7, line8, line9

anim = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=True)
plt.tight_layout()
plt.show()