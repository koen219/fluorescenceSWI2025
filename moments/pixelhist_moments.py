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
    img = np.zeros((data.shape[0],data.shape[1],4))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            photon_count = np.sum(data[i,j,:])
            if photon_count > 50:
                cutoff = np.min(np.nonzero(data[i,j,:]))
                shifted_bins = data[i,j,cutoff:]
                N_bins = data.shape[2]
                
                m1 = np.sum(np.multiply(shifted_bins, times[:N_bins-cutoff])) / photon_count
                m2 = np.sum(np.multiply(shifted_bins, times[:N_bins-cutoff]**2)) / photon_count
                m3 = np.sum(np.multiply(shifted_bins, times[:N_bins-cutoff]**3)) / photon_count
                (t1,t2,p) = util.params_from_moments(m1,m2,m3)
                a = 6 * (2*(m1**2) - m2)
                b = 2 * (m3 - 3 * m1*m2)
                c = 3*(m2**2) - 2*m1*m3

                D = b**2 - 4*a*c
                if a != 0 and D > 0:
                    r1 = (-b - np.sqrt(D)) / (2 * a)
                    r2 = (-b + np.sqrt(D)) / (2 * a)
                    p = (m1 - r2) / (r1 - r2)
                    
                    img[i,j,0] = p
                    img[i,j,1] = r1
                    img[i,j,2] = r2
                    img[i,j,3] = 1

    flat_p = img[:,:,0].flatten()
    flat_r1 = img[:,:,1].flatten()
    flat_r2 = img[:,:,2].flatten()

    flat_p = flat_p[np.all([flat_p > 0, flat_p < 1],axis=0)]
    flat_r1 = flat_r1[np.all([flat_r1 > 0, flat_r1 < 10],axis=0)]
    flat_r2 = flat_r2[np.all([flat_r2 > 0, flat_r2 < 5],axis=0)]


    plt.hist(flat_r1,bins=100)
    plt.title("Method of moments, pixel histogram")
    plt.xlabel("Slow decay time (ns)")
    plt.ylabel("Counts")
    plt.savefig('img/moments_slow_decay_pixel_histogram.png')
    plt.show()
    plt.hist(flat_r2,bins=100)
    plt.title("Method of moments, pixel histogram")
    plt.xlabel("Fast decay time (ns)")
    plt.ylabel("Counts")
    plt.savefig('img/moments_fast_decay_pixel_histogram.png')
    plt.show()

def load_data():
    # index 0: x position
    # index 1: y position
    # index 3: time
    # numerical value: number of photons measured at (x,y) in time bin t
    # time bins range from 0ns to 25ns
    data = np.load("./best_fit.npy")
    return data[:, :, 0, :, 0, 0,]

if __name__ == '__main__':
    main()