import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
Gx, Gy = 40, 60  # Grid size
T = 50  # Number of time frames
K = 3   # Number of dyes

# Define lifetimes for each dye
#taus = np.random.uniform(5, 20, K)  # Fluorescence lifetimes between 5 and 20 frames
taus = np.array([3,6,8])
lambdas = np.random.uniform(80, 120, K)  # Initial photon intensity
print(taus)
print(lambdas)

# Generate spatial masks for each dye (ensuring they don’t overlap perfectly)
object_masks = np.zeros((K, Gx, Gy))
for i in range(K):
    x0, y0 = np.random.randint(10, 40), np.random.randint(10, 40)  # Random center
    radius = np.random.randint(5, 10)  # Random radius
    x, y = np.meshgrid(np.arange(Gy), np.arange(Gx))
    object_masks[i] = ((x - x0) ** 2 + (y - y0) ** 2) < radius ** 2  # Circular object

# Initialize photon count at t=0
initial_photons = np.zeros((K, Gx, Gy))  # Now track each dye separately
for i in range(K):
    initial_photons[i] = lambdas[i] * object_masks[i]  # Assign photons per dye

# Prepare the decay data array
decay_data = np.zeros((Gx, Gy, T))

# Simulate decay process
for t in range(T):
    total_photons = np.zeros((Gx, Gy))  # Reset total count for this time step
    
    for i in range(K):
        # Apply independent exponential decay per dye
        photons_t = initial_photons[i] * np.exp(-t / taus[i])
        total_photons += photons_t  # Sum contributions from all dyes
    
    # Ensure values stay within a reasonable range before Poisson sampling
    lam_safe = np.clip(total_photons, 0, 1e3)  # Prevent overflow in Poisson
    decay_data[:, :, t] = np.random.poisson(lam_safe)

# Visualization Function
fig, ax = plt.subplots()
im = ax.imshow(decay_data[:, :, 0], cmap="inferno", vmin=0, vmax=np.max(decay_data))

def update(frame):
    im.set_array(decay_data[:, :, frame])
    ax.set_title(f"Photon Decay at t={frame}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=T, interval=200, blit=False)
plt.show()

# Save data for further analysis
print(np.max(decay_data))
np.save("photon_decay_data.npy", decay_data)