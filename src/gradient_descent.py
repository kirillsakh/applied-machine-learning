import numpy as np

class LinearRegression():

    def __init__(self, alpha=0.1, n_iterations=1000):

        self.alpha = alpha
        self.n_iterations = n_iterations

    def _normalize_matrix(self, X):
        return (X - np.mean(X, 0)) / np.std(X, 0)

    def _extend_matrix(self, X):
        return np.hstack([ np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):

        self.weights = np.random.randn(len(X[0]) + 1)*0.01
        X = self._normalize_matrix(X)
        X = self._extend_matrix(X)

        for i in range(self.n_iterations):

            y_predicted = X @ self.weights
            residue = y_predicted - y
            gradient = X.T @ residue
            self.weights -= (self.alpha/len(X)) * gradient

        return self

    def predict(self, X):

        X = self._normalize_matrix(X)
        X = self._extend_matrix(X)
        return X @ self.weights

    def multiply(self, a, b):
        '''
        multiply: multiply matrix 'A' and matrix 'B',
        much slower compared to Numpy/C implementation
        '''

        result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
        result = np.asarray(result, dtype=np.float64)

        arows = len(a)
        brows = len(b)
        bcols = len(b[0])

        for i in range(arows):
            for j in range(bcols):
                for k in range(brows):
                    result[i][j] += a[i][k] * b[k][j]

        return result
