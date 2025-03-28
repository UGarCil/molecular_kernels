'''An implementation of local gaussian kernel for molecular similarity computation.'''
import numpy as np
# from numba import njit, prange

# FD. pyget_local_kernels_gaussian
# purp. Compute Gaussian kernels between local atomic environments.
def pyget_local_kernels_gaussian(q1:np.array, q2:np.array, n1:np.array, n2:np.array, sigmas:np.array) -> np.array:
    """
    Computation of gaussian kernels between local atomic environments.
    
    Parameters:
    -----------
    q1 : ndarray (n_features, n_atoms1)
        Atomic descriptors for molecule set 1
    q2 : ndarray (n_features, n_atoms2)
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
    # Get number of molecules and sigmas
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(sigmas)
    
    # Precompute -1.0 / (2 * sigma^2)
    inv_sigma2 = -0.5 / (sigmas ** 2)
    
    # Compute start indices for each molecule
    i_starts = np.cumsum(n1) - n1
    j_starts = np.cumsum(n2) - n2
    
    # Initialize kernel array
    kernels = np.zeros((nsigmas, nm1, nm2))
    
    # Compute kernels
    _compute_kernels(q1, q2, n1, n2, inv_sigma2, i_starts, j_starts, kernels)
    
    return kernels

# @njit(parallel=True)
def _compute_kernels(q1, q2, n1, n2, inv_sigma2, i_starts, j_starts, kernels):
    '''
    Compute Gaussian kernels for all sigmas.
    '''
    nm1 = len(n1)
    nm2 = len(n2)
    nsigmas = len(inv_sigma2)
    
    max_n1 = np.max(n1)
    max_n2 = np.max(n2)
    
    for a in range(nm1):
        for b in range(nm2):
            ni = n1[a]
            nj = n2[b]
            
            # Compute pairwise distances
            atomic_distance = np.zeros((max_n1, max_n2))
            for i in range(ni):
                for j in range(nj):
                    diff = q1[:, i + i_starts[a]] - q2[:, j + j_starts[b]]
                    atomic_distance[i, j] = np.sum(diff ** 2)
            
            # Compute Gaussian kernels for all sigmas
            for k in range(nsigmas):
                total = 0.0
                for i in range(ni):
                    for j in range(nj):
                        total += np.exp(atomic_distance[i, j] * inv_sigma2[k])
                kernels[k, a, b] = total
                


# EXAMPLE IMPLEMENTATION
# Atomic descriptors (3D features in this example)
# Each row corresponds to a different atom's features
# Each column corresponds to a different atom in the molecule
# Example: H2O, HF, H2O (3D coordinates for 3 atoms each)
q1 = np.array([[-0.8, 0.4, 0.4, -0.6, 0.6, -0.8, 0.4, 0.4],
               [-0.8, 0.4, 0.4, -0.6, 0.6, -0.8, 0.4, 0.4],
               [-0.8, 0.4, 0.4, -0.6, 0.6, -0.8, 0.4, 0.4]])  # H2O, HF, H2O sequentially

# Number of atoms per molecule
n1 = np.array([3, 2, 3])  # First H2O (3), HF (2), Second H2O (3)

# Second set of atomic descriptors (can be the same or different)
# # Another set of molecules (can be the same or different)
# q2 = np.array([[-0.8, 0.4, 0.4, -0.6, 0.6]])  # H2O, HF sequentially
# n2 = np.array([3, 2])  # One H2O (3) and one HF (2)

# Kernel width (sigma values)
sigmas = np.array([0.5])  # Example Gaussian width



pyget_local_kernels_gaussian(q1, q1, n1, n1, sigmas)

# Notes on implementation:
'''
The prgram receives as inputs information on the molecules;
The features of such molecules per atom are stored in q1 and q2;
In the scenario where more than one molecule has to be represented in a single array,
the process happens by stacking multiple molecules in the same array.
The second parameter set, n1 and n2, are the number of atoms per molecule.

For example, if the goal is to compare the molecules
H2O, HF, and H2O by using as features the electric charge (a 1D array),
the input q1 would be:
q1 = np.array([[-0.8, 0.4, 0.4, -0.6, 0.6, -0.8, 0.4, 0.4]]) # H2O, HF, H2O sequentially
                 H     O    F     H    O     H    O   F
n1 = [3, 2, 3] # First H2O (3), HF (2), Second H2O (3)

The second set of atomic descriptors (can be the same or different)

Assumptions:
- p1 and p2 represent features in a molecule set, where each column is an atom
- p1 and p2 are square (2D) arrays, with shape (n_features, n_atoms)
- The sum of elements in n1 equals the number of columns in p1

'''