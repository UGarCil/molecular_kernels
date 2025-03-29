import numpy as np
from numba import njit, prange

def pyget_vector_kernels_gaussian(q1: np.ndarray, q2: np.ndarray, 
                                n1: np.ndarray, n2: np.ndarray, 
                                sigmas: np.ndarray) -> np.ndarray:
    """
    Compute Gaussian kernels between molecules with vectorized atomic environments.
    
    Parameters:
    -----------
    q1 : ndarray (nm1, max_n1, n_features)
        Atomic descriptors for molecule set 1
    q2 : ndarray (nm2, max_n2, n_features)
        Atomic descriptors for molecule set 2
    n1 : ndarray (nm1,)
        Number of atoms for each molecule in set 1
    n2 : ndarray (nm2,)
        Number of atoms for each molecule in set 2
    sigmas : ndarray (nsigmas,)
        Gaussian kernel width parameters
        
    Returns:
    --------
    kernels : ndarray (nsigmas, nm1, nm2)
        Computed Gaussian kernels
    """
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(sigmas)
    
    # Precompute -1.0 / (2 * sigma^2)
    inv_sigma2 = -0.5 / (sigmas ** 2)
    
    # Initialize kernel array
    kernels = np.zeros((nsigmas, nm1, nm2))
    
    # Call the optimized computation function
    _compute_vector_kernels(q1, q2, n1, n2, inv_sigma2, kernels)
    
    return kernels

@njit(parallel=True)
def _compute_vector_kernels(q1, q2, n1, n2, inv_sigma2, kernels):
    """
    Numba-optimized kernel computation with parallel processing.
    """
    nm1 = q1.shape[0]
    nm2 = q2.shape[0]
    nsigmas = len(inv_sigma2)
    max_n1 = q1.shape[1]
    max_n2 = q2.shape[1]
    
    # Preallocate atomic distance matrix
    atomic_distance = np.zeros((max_n1, max_n2))
    
    for i in prange(nm1):
        for j in range(nm2):
            ni = n1[i]
            nj = n2[j]
            
            # Reset distance matrix for this pair
            atomic_distance[:, :] = 0.0
            
            # Compute pairwise squared distances
            for ia in range(ni):
                for ja in range(nj):
                    diff = q1[i, ia] - q2[j, ja]
                    atomic_distance[ia, ja] = np.sum(diff ** 2)
            
            # Compute Gaussian kernels for all sigmas
            for k in range(nsigmas):
                total = 0.0
                for ia in range(ni):
                    for ja in range(nj):
                        total += np.exp(atomic_distance[ia, ja] * inv_sigma2[k])
                kernels[k, i, j] = total
                
# Example data (3 molecules, max 4 atoms, 3 features)
q1 = np.random.rand(3, 4, 3)  # 3 molecules, max 4 atoms, 3 features
q2 = np.random.rand(2, 4, 3)  # 2 molecules, max 4 atoms, 3 features
n1 = np.array([3, 2, 4])      # Atoms per molecule in q1
n2 = np.array([4, 3])         # Atoms per molecule in q2
sigmas = np.array([0.5, 1.0]) # Kernel widths

kernels = pyget_vector_kernels_gaussian(q1, q2, n1, n2, sigmas)


'''
Notes on the implementation:
Each molecule has up to max_n atoms, and each atom has n_features features.
Given a dataset structure of (number molecules, max num atoms, num features),
the representation of a single molecule H2O in a dataset where the maximum number 
of atoms is 4 and the number of features is 1, would be:

[-4,2,-4,0]

here, each column corresponds to an atom in the molecule
Each row is a feature of the atom.
The position is zero padded to the right, so that the maximum number of atoms is 4.

The dimensions of the inputs would become 2D if the number of features is > 1.
The dimensions of the inputs would become 3D if the number of features is > 1 AND the number 
of molecules is > 1.

'''