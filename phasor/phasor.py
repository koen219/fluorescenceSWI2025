import numpy as np
import matplotlib.pyplot as plt
import itertools


def load_data():
    """
    This function is used to prepare the data for the remainder of the script.
    This function should return a numpy array of dimensions: width x height x time aswell as the time points of each bin. The data file used for this project is not included.
    """
    data = np.load("./best_fit.npy")[:, :, 0, :, 0, 0]
    dt = 0.1  # This was true for the best_fit.npy datafile.
    bins = data.shape[2]
    max_time = bins * dt
    times = np.linspace(0, max_time, num=bins + 1)

    return (data, times)

def load_simulated_data():
    data = np.load("photon_decay_data.npy") 
    dt = 0.1  # This was true for the best_fit.npy datafile.
    bins = data.shape[2]
    max_time = bins * dt
    times = np.linspace(0, max_time, num=bins + 1)
    return data,times

def make_simulated_data(tau1, tau2, rho, N, number_of_pixels):
    """
    This function creates simulated data and outputs it in the same form as the real data. This way it can be used to test the scripts.

    Args:
        tau1 (float): Decay time of dye 1
        tau2 (float): Decay time of dye 2
        rho (float|None): Mixing proportion of dyes, or None if this should be random
        N (int): The number of photons for each pixel
        number_of_pixels (int): Number of pixels

    Returns:
        Same as load_data() except with spatially not correlated data.
    """
    if rho == None:
        rho_factory = lambda: np.random.uniform(0,1)
    else: # If rho
        rho_factory = lambda: rho

    max_time = 100
    bins = 250

    times = np.linspace(0, max_time, num=bins + 1)
    output = np.zeros((number_of_pixels, 1, bins))

    for k in range(number_of_pixels):
        rho = rho_factory()

        N1 = int(rho * N)
        N2 = int(N - N1)

        # Generate exponential samples
        dist1 = np.random.exponential(scale=tau1, size=N1)
        dist2 = np.random.exponential(scale=tau2, size=N2)
        dist = np.concatenate([dist1,dist2])
        dist = dist[dist < max_time]

        # Compute histogram
        hist, _ = np.histogram(dist, bins=times)

        output[k, 0, :] = hist
    
    return output, times

def compute_phasors(data, times, omega, photon_cutoff):
    """
        Function computes the phasors of the input at specified omega. 

        Args:
            data: data loaded as in the load_data function.
            times: times as computed in load_data function
            omega (float): Frequency used in the transform.
            photon_cutoff (int): Minimum number of photons present in a pixel for this pixel to be used in the computation.

        Returns:
            A dictonary with keys the pixel values and values the phasor for that pixel.
    """

    output = dict()
    for (i,j) in itertools.product(range(data.shape[0]), range(data.shape[1])):
        
        # We might have to shift the data if the peak is not exactly at t = 0.
        # This step is why we write it in a slow for loop.
        time_cutoff_index = np.argmax( data[i,j,:])
        num_bins = data.shape[2] - time_cutoff_index
        time_to_use = times[:num_bins]

        dist = data[i,j, time_cutoff_index:].copy() # Don't want to change data inplace!

        # If this pixel has fewer than cutoff photons discard it.
        if np.sum(dist) <= photon_cutoff:
            continue
        dist /= np.sum(dist)

        phi = np.sum( dist * np.exp(1.0j * omega * time_to_use))
        output[(i,j)] = phi
    
    return output

def plot_phasors(phasors):
    """Create a plot of phasors

    Args:
        phasors (list[complex]): List of phasors

    Returns:
        fig, ax on which the phasors are plotted.
    """
    fig, ax = plt.subplots()

    x = [phi.real for phi in phasors ]
    y = [phi.imag for phi in phasors ]
    ax.scatter(x,y)

    t = np.linspace(0, np.pi, num=100) 
    x = 0.5 * np.cos(t) + 0.5
    y = 0.5 * np.sin(t)
    ax.plot(x, y, 'r--')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.0)

    return fig, ax


def compute_taus(data,times, omega):
    """  Compue the decay times tau1 and tau2 by fitting a line through the phasor plot
    
    Args:
         data: data loaded as in the load_data function.
         times: times as computed in load_data function

    """ 
    def solve_quadratic(a,b,c):
         D = b**2 - 4*a*c
         return ((-b + np.sqrt(D)) / (2*a), 
                 (-b - np.sqrt(D)) / (2*a))

    phasors = np.array(list(compute_phasors(data, times, omega, 50).values())).view(float).reshape(-1,2)
    # Fit line through the data and compute the intersection with the circle (x - 0.5)^2 + y^2 = (1/2)^2. 
    m, b = np.polyfit(phasors[:,0], phasors[:,1], 1)
    (x1, x2) = solve_quadratic(1 + m**2, 2*m*b - 1, b**2)
    y1 = m*x1 + b
    y2 = m*x2 + b
    z1 = complex(x1,y1)
    z2 = complex(x2,y2)
    # Use that phi = 1 / (1 + i * omega * tau ) for pure exponentials
    taus = ( (1/omega)* z1.imag / z1.real,  (1/omega) * z2.imag / z2.real)

    return taus

def compute_rhos(data, times, omega, photon_cutoff):
    """
        Function that determines the fraction of dye1 in each pixel.

        Args:
            data: data loaded as in the load_data function.
            times: times as computed in load_data function
            omega (float): Frequency used by the phasor method.
            photon_cutoff (int): Minimum number of photons present in a pixel for this pixel to be used in the computation.

        Returns:
            A dictonary with keys the pixel values and as values the dye proportion.
    """
    phasors = compute_phasors(data,times, omega,photon_cutoff)
    taus = compute_taus(data, times, omega)
    
    phi1 = 1.0 / (1.0 + taus[0] * omega * 1.0j)
    phi2 = 1.0 / (1.0 + taus[1] * omega * 1.0j)

    # Rhos values per pixel
    rhos = dict()

    for pixel, phi in phasors.items():
        # Compute projection along the line trough phi1 and phi2. 

        # First shift with phi1
        vhat = phi - phi1
        v = phi2 - phi1
        
        # Compute projection and translate back.
        numerator = vhat.real * v.real + vhat.imag * vhat.imag
        denominator = v.real*v.real + v.imag*v.imag
        rho = numerator / denominator

        rhos[pixel] =rho

    return rhos

def make_pictures_based_on_rho_and_intensity(data,times):
    """ Script used for presentation pictures. These don't really make sense with simulated data.
    """
    omega = 0.1

    rhos = np.zeros(( data.shape[0], data.shape[1] ))
    for pixel, value in compute_rhos(data,times, 0.1, 50).items():
        rhos[*pixel] = value

    intensity = np.sum(data,axis=2)

    image = np.zeros( (intensity.shape[0], intensity.shape[1], 2) )
    image[:,:,0] = intensity / np.sum(intensity)
    image[:,:,1] = rhos
    
    mean_rho = np.mean(rhos[rhos >0])

    # Determine global min and max for consistent color scaling
    vmin = image[:, :, 0].min()  # Minimum intensity value
    vmax = image[:, :, 0].max()  # Maximum intensity value

    plt.imshow(image[:,:,0], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f"All_tau_figure_intensity_and_rho_omega_{omega}.jpg", dpi=300)
    plt.close()
    plt.figure()

    image_high = image.copy()
    image_high[image[:, :, 1] < mean_rho] = 0
    plt.imshow(image_high[:,:,0], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f"High_tau_figure_intensity_and_rho_omega_{omega}.jpg", dpi=300)
    plt.close()

    image_low = image.copy()
    image_low[image[:, :, 1] >= mean_rho] = 0
    plt.imshow(image_low[:,:,0], vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.savefig(f"Low_tau_figure_intensity_and_rho_omega_{omega}.jpg", dpi=300)
    plt.close()
    # plt.show()

if __name__ == '__main__':
    data, times = load_data()
    data, times = load_simulated_data()
    # data, times = make_simulated_data(1.0, 2.0, None, 100, 500)
    phasors = compute_phasors(data, times, 0.25, 50)
    taus = compute_taus(data,times, 0.1)
    print(f"Computed taus = {taus}")
    compute_rhos(data, times, 0.1, 50)
    make_pictures_based_on_rho_and_intensity(data,times)
    plot_phasors(phasors.values())
    plt.show()
    
    
