# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:46:40 2025

@author: joriswinden
"""

import numpy as np
import matplotlib.pyplot as plt

import util

def main():
    data = load_data()

    times = np.linspace(0,25,data.shape[2])

    # get first three moments by combining all pixel data
    (m1,m2,m3) = moments_from_data(data,times)

    # calculate (global) parameters from all pixel data
    (t1,t2,p) = util.params_from_moments(m1,m2,m3)

    # segment pixels using global t1 and t2
    stats = pixel_stats(data,times,t1,t2)

    # segment by multiplying pixel count by p
    img1 = np.multiply(stats[:,:,0],stats[:,:,1])
    img2 = np.multiply(stats[:,:,0],1-stats[:,:,1])

    print(f"Estimated slow lifetime: {t1:.1f} ns")
    print(f"Estimated fast lifetime: {t2:.1f} ns")
            
    plt.imshow(img1)
    plt.title(r'Slow dye, $\tau \approx 3.5$ ns')
    plt.axis('off')
    plt.savefig('img/moments_2step_img_fastdye',bbox_inches='tight')
    plt.show()
    plt.imshow(img2)
    plt.title(r'Fast dye, $\tau \approx 1.4$ ns')
    plt.axis('off')
    plt.savefig('img/moments_2step_img_slowdye',bbox_inches='tight')
    plt.show()

def load_data():
    # index 0: x position
    # index 1: y position
    # index 3: time
    # numerical value: number of photons measured at (x,y) in time bin t
    # time bins range from 0ns to 25ns
    data = np.load("./best_fit.npy")
    return data[:, :, 0, :, 0, 0,]

def moments_from_data(data,times):
    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    count = 0
    N_bins = data.shape[2]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            photon_count = np.sum(data[i,j,:])
            if photon_count > 25:
                # fix time shift by identifying first photon
                cutoff = np.min(np.nonzero(data[i,j,:]))
                shifted_data = data[i,j,cutoff:]
                
                m1 = m1 + np.sum(np.multiply(shifted_data, times[:N_bins-cutoff]))
                m2 = m2 + np.sum(np.multiply(shifted_data, times[:N_bins-cutoff]**2))
                m3 = m3 + np.sum(np.multiply(shifted_data, times[:N_bins-cutoff]**3))
                count = count + photon_count
    m1 = m1 / count
    m2 = m2 / count
    m3 = m3 / count
    return (m1,m2,m3)

def pixel_stats(data,times,t1,t2):
    stats = np.zeros((data.shape[0],data.shape[1],3))
    N_bins = data.shape[2]
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            photon_count = np.sum(data[i,j,:])
            if photon_count > 25:
                # fix time shift by identifying first photon
                cutoff = np.min(np.nonzero(data[i,j,:]))
                shifted_data = data[i,j,cutoff:]
                
                mean = np.sum(np.multiply(shifted_data, times[:N_bins-cutoff])) / photon_count
                p = (mean - t2) / (t1 - t2)
                # # fix when p might fall outside of [0,1]
                p = max(min(p,1),0)
                    
                stats[i,j,:] = (photon_count,p,mean)
    return stats

if __name__ == '__main__':
    main()