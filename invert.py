

import numpy as np
from scipy.spatial.transform import Rotation as R

def invert_33_eigs(A):
    """
    Invert a 3x3 matrix using its eigenvalues and eigenvectors.
    """
    # Compute the eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Invert the eigenvalues
    inverted_eigenvalues = 1 / eigenvalues

    # Construct the inverted matrix using the eigenvectors and inverted eigenvalues
    A_inv = eigenvectors @ np.diag(inverted_eigenvalues) @ eigenvectors.T
    
    return A_inv

# Example usage:
A = np.array([[4, 0, 0],
              [0, 9, 0],
              [0, 0, 1]])

A_inv = invert_33_eigs(A)
print(A_inv)

print(np.linalg.inv(A))



def cov_from_scales(rotation:np.ndarray, scales:np.ndarray):
    """
    Compute the covariance matrix from the rotation and scales.
    """
    return rotation @ np.diag(scales) @ rotation.T


def inv_cov_from_scales(rotation:np.ndarray, scales:np.ndarray):
    """
    Compute the inverse covariance matrix from the rotation and scales.
    """
    return rotation @ np.diag(1/scales) @ rotation.T


# random 3x3 orthogonal matrix
m = R.random().as_matrix()

# random scales
scales = np.random.rand(3)

# compute the covariance matrix
cov = cov_from_scales(m, scales)

# compute the inverse covariance matrix
inv_cov = inv_cov_from_scales(m, scales)

print(cov @ inv_cov) 
