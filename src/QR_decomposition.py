import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None

    def _extend_matrix(self, X):
        return np.hstack([ np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        X = self._extend_matrix(X)
        Q, R = qr_decomposition(X)
        self.weights = solve_triangular(R, Q.T @ y)
    
    def predict(self, X):
        X = self._extend_matrix(X)
        return X @ self.weights

def qr_decomposition(A):
    '''
    Given an n x m invertable matrix 'A', returns
    a decomposition of 'A' into a product A = QR of
    an orthogonal n x m matrix 'Q' and
    an upper triangular m x m matrix 'R'
    '''
    
    n, m = A.shape
    Q = np.identity(n)
    R = A.copy()
    
    for i in range(m):
        v = R[i:, i]
            
        e = np.zeros(n-i)
        e[0] = 1  
        
        H = np.identity(n)
        H[i:, i:] = householder_transformation(v, e)

        Q = Q @ H.T
        R = H @ R
    
    return Q, R

def householder_transformation(a, e):
    '''
    Given 'a' vector and a unit vector 'e',
    returns an orthogonal matrix a.k.a reflection
    that transforms 'a' into a direction
    parallel to the line of 'e'
    '''
    
    u = a - np.sign(a[0]) * np.linalg.norm(a) * e  
    v = u / np.linalg.norm(u)
    H = np.identity(len(a)) - 2 * np.outer(v, v)
    
    return H

def solve_triangular(A, b):
    '''
    Solves the equation Ax = b when 'A'
    is an upper-triangular square matrix
    and 'b' is a one dimensional vector
    by back-substitution. The length of 'b'
    and the number of rows must match.
    Returns 'x' as a one-dimensional np.ndarray
    of the same length as 'b'.
    '''

    _, m = A.shape
    x = np.zeros(m)
    for i in range(m-1, -1, -1):
        tmp = b[i]
        for j in range(m-1, i, -1):
            tmp -= np.dot(A[i,j],x[j])
        x[i] = tmp / A[i,i]
  
    return x

