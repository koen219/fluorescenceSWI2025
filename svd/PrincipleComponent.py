import numpy as np
import matplotlib.pyplot as plt

intensity = np.load("photon_decay_data.npy")
(x,y,q) = intensity.shape
intensity_m = np.array([intensity[:,:,i].flatten() for i in range(q)]).T # Flatten intenisty array (easier for post-processing)
n = intensity_m.shape[0]

intensity_m_bar = np.zeros(q)
intensity_p = np.zeros((n,q))
for i in range(q):
    intensity_m_bar[i] = np.average(intensity_m[:,i])
    for k in range(n):
        intensity_p[k,i] =  intensity_m[k,i]/np.sqrt(intensity_m_bar[i]) # Rescale for Poisson noise correction

C = np.zeros((q,q))
intensity_p_bar = np.array([np.average(intensity_p[:,i]) for i in range(q)])
for i in range(q):
    for j in range(i,q):
        C[i,j] = np.sum([(intensity_p[k,i]-intensity_p_bar[i])*(intensity_p[k,j]-intensity_p_bar[j]) for k in range(n)])/n
C = C + C.T - np.diag([C[i][i] for i in range(q)]) # Save computation power by taking into account symmetry of matrix

U, D, Vh = np.linalg.svd(C, hermitian=True) # Symmetric matrix, so hermitian (real valued)
S = (Vh @ intensity_p.T).T
for i in range(q):
    if np.average(S[:,i]) < 0:
        if np.abs(np.max(S[:,i])) < np.abs(np.min(S[:,i])):
            Vh[i,:] = -1*Vh[i,:] # Eigenvectors are not unique, we want the sign of the vector such that the object with the highest absolute intensity is postive

Vh[2,:] = -1*Vh[2,:] # Hard coding for this specific image, because if-statement before did not do its job

S = (Vh @ intensity_p.T).T
S = S.reshape(x,y,q) # Undo the flattening done at the beginning
for i in range(q):
    rho_x = np.corrcoef(np.array([S[:,0:x-1,i].flatten(),S[:,1:x,i].flatten()])) # Calculate Pearson's correlation coefficients
    rho_y = np.corrcoef(np.array([S[0:x-1,:,i].flatten(),S[1:x,:,i].flatten()]))
    alpha = (rho_x[0][1]+rho_y[0][1])/2
    if alpha>=0.1: # Threshold on Pearson's correlation coefficient to determine if we're dealing with a prinicipal component or noise
        print(i)
        print(alpha)

'''
plt.subplot(2,2,1)
plt.title("Inner Product with First Right Eigenvector")
plt.imshow(S[:,:,0])
plt.colorbar()
plt.subplot(2,2,2)
plt.title("Inner Product with Second Right Eigenvector")
plt.imshow(S[:,:,1])
plt.colorbar()
plt.subplot(2,2,3)
plt.title("Inner Product with Third Right Eigenvector")
plt.imshow(S[:,:,2])
plt.colorbar()
plt.subplot(2,2,4)
plt.title("Structure of Squared Right Eigenvectors")
plt.plot(np.arange(q),Vh[0,:]**2, label="1st EV")
plt.plot(np.arange(q),Vh[1,:]**2, label="2nd EV")
plt.plot(np.arange(q),Vh[2,:]**2, label="3rd EV")
plt.plot(np.arange(q),1/3*np.exp(-np.arange(q)/3), label="Exp(3)")
plt.plot(np.arange(q),1/6*np.exp(-np.arange(q)/6), label="Exp(6)")
plt.plot(np.arange(q),1/8*np.exp(-np.arange(q)/8), label="Exp(8)")
plt.legend()
plt.xlabel("time")
'''

plt.subplot(1,3,1)
plt.title("Inner Product with First Right Eigenvector")
plt.imshow(S[:,:,0])
plt.colorbar(shrink=0.28)
plt.subplot(1,3,2)
plt.title("Inner Product with Second Right Eigenvector")
plt.imshow(S[:,:,1])
plt.colorbar(shrink=0.28)
plt.subplot(1,3,3)
plt.title("Inner Product with Third Right Eigenvector")
plt.imshow(S[:,:,2])
plt.colorbar(shrink=0.28)
plt.savefig("PrincipalComponent.pdf", format="pdf")
plt.show()