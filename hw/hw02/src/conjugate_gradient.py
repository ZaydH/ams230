import numpy as np

from typing import List


class ConjugateGradient:
    def __init__(self):
        self._A = None
        self._n = None
        self.n = None

        self._x0 = None
        self._b = None

        self._tol = 10 ** -6

    def initialize_x0(self):
        """
        Initialize x0 to 2*rand(n,1) as specified in the homework description.
        """
        assert self.n is not None
        self._x0 = 2 * np.random.rand([self.n, 1])

    def initialize_b(self):
        """
        Initialize the offset vector b from the f(x) = x^T * A * x - b * x
        """
        assert self.n is not None
        self._b = np.random.rand([self.n, 1])


    def initialize_matrix(self, clusters: List[List[float]], p: List[float] = None):
        """
        Initializes the matrix A for the conjugate gradient.

        :param p: Probability of selecting each cluster.
        :param clusters: Cluster ranges for the eigenvalues.
        """
        assert self.n is not None
        assert (p is None and len(clusters) == 1) or (len(p) == len(clusters))

        eigenvals = []
        for _ in range(n):
            if p is not None:
                i = np.argmax(np.random.multinomial(1, p)) # type: int
            else:
                i = 0
            eigenvals.append(np.random.uniform(clusters[i][0], clusters[i][1]))

        # As explained in the homework description, initialize the matrix using QR decomposition
        # and the specified eigenvalues.
        q = np.linalg.qr(np.random.rand(self.n, self.n))
        diag = np.diag(eigenvals, 1)
        self._A = q.transpose() * diag * q

    def run(self):
        """
        Perform the conjugate gradient test.
        """
        x_k = self._x0
        r_k = self._A * x_k - self._b  # type: np.ndarray
        p_k = -1 * r_k
        k = 0

        while k < self.n and np.linalg.norm(r_k) < self._tol:
            alpha_k = (r_k.transpose() * r_k) / (p_k.transpose() * self._A * p_k)
            r_k_1 = r_k + alpha_k * self._A * p_k

            beta_k_1 = (r_k_1.transpose() * r_k_1) / (r_k.transpose() * r_k)

            # Increment the counter and vectors
            k = k + 1
            x_k = x_k + alpha_k * p_k
            r_k = r_k_1
            p_k = -r_k_1 + beta_k_1 * p_k



def run_cg(n, clusters, p=None):
    cg = ConjugateGradient()
    cg.n = n

    cg.initialize_x0()
    cg.initialize_b()

    if p is None:
        cg.initialize_matrix(clusters)
    else:
        cg.initialize_matrix(clusters, p)


def main():



if __name__ == "__main__":
    main(1000, [10, 1000])
    main(1000, [10, 1000])
