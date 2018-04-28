import math
import numpy as np
from typing import List

from cg_utils import build_plot, setup_logger


class ConjugateGradient:
    def __init__(self):
        self.n = None

        self._x0 = None

        self._A = None
        self._b = None
        self._x_opt = None
        self._err_opt = None

        self._err = None

        self._tol = 10 ** -15

    def initialize_x0(self):
        """
        Initialize x0 to 2*rand(n,1) as specified in the homework description.
        """
        assert self.n is not None
        self._x0 = 2 * np.random.rand(self.n, 1)

    def initialize_b(self):
        """
        Initialize the offset vector b from the f(x) = x^T * A * x - b * x
        """
        assert self.n is not None
        self._b = np.random.rand(self.n, 1)

        if self._A is not None:
            self._find_x_opt()

    def initialize_matrix(self, clusters: List[List[float]], p: List[float] = None):
        """
        Initializes the matrix A for the conjugate gradient.

        :param p: Probability of selecting each cluster.
        :param clusters: Cluster ranges for the eigenvalues.
        """
        assert self.n is not None
        assert (p is None and len(clusters) == 1) or (len(p) == len(clusters))

        eigenvals = []
        for _ in range(self.n):
            if p is not None:
                i = np.argmax(np.random.multinomial(1, p))  # type: int
            else:
                i = 0
            eigenvals.append(np.random.uniform(clusters[i][0], clusters[i][1]))

        # As explained in the homework description, initialize the matrix using QR decomposition
        # and the specified eigenvalues.
        q, r = np.linalg.qr(np.random.rand(self.n, self.n))
        diag = np.diag(eigenvals)
        self._A = q.transpose() @ diag @ q

        if self._b is not None:
            self._find_x_opt()

    def run(self, title):
        """
        Perform the conjugate gradient experiment.
        """
        self._err = []

        x_k = self._x0
        r_k = self._A @ x_k - self._b  # type: np.ndarray
        p_k = -1 * r_k
        k = 0
        self.update_error(x_k)

        while k < self.n and np.linalg.norm(r_k, 2) > self._tol:
            alpha_k = (r_k.transpose() @ r_k) / (p_k.transpose() @ self._A @ p_k)

            x_k = x_k + alpha_k * p_k
            r_k_1 = r_k + alpha_k * self._A @ p_k

            beta_k_1 = (r_k_1.transpose() @ r_k_1) / (r_k.transpose() @ r_k)

            # Increment the counter and vectors
            k = k + 1
            p_k = -r_k_1 + beta_k_1 * p_k
            r_k = r_k_1
            self.update_error(x_k)

        build_plot(list(range(0, len(self._err))), self._err, x_label="Iteration",
                   y_label="Log Error", title=title)

    def update_error(self, x_k: np.ndarray):
        """
        Calculate the error for x_k and update the error list.

        :param x_k: x-value for iteration k,
        """
        diff = abs(self._calculate_err(x_k) - self._err_opt)
        if diff == 0:
            return
        self._err.append(math.log10(diff))

    def _calculate_err(self, x: np.ndarray) -> float:
        """
        Calculate the error for the specified x for the convex quadratic function.

        :param x: X position to be passed to the error function.
        :return: Error with respect to the convex quadratic function.
        """
        err = 0.5 * (x.transpose() @ self._A @ x)  # "@" symbol in Python 3.5 is matrix multiply
        err -= self._b.transpose() @ x
        return err[0]

    def _find_x_opt(self):
        """
        Finds the optimal value that minimizes the convex function.
        """
        self._x_opt = np.linalg.solve(self._A, self._b)

        self._err_opt = self._calculate_err(self._x_opt)


def run_cg(n, clusters, title, p=None):
    cg = ConjugateGradient()
    cg.n = n

    cg.initialize_x0()
    cg.initialize_b()

    if p is None:
        cg.initialize_matrix(clusters)
    else:
        cg.initialize_matrix(clusters, p)

    cg.run(title)


def main():
    run_cg(10, [[10, 1000]], "uniform_eigen_10_1000")
    run_cg(1000, [[9, 11], [999, 1001]], "dual_modal_eigen_cluster_10_1000", [.1, .9])


if __name__ == "__main__":
    setup_logger()
    main()
