import numpy as np
import matplotlib.pyplot as plt

from typing import List


class ConjugateGradient:
    def __init__(self):
        self._n = None
        self.n = None

        self._x0 = None

        self._A = None
        self._b = None
        self._x_opt = None
        self._err_opt = None

        self._err = None

        self._tol = 10 ** -6

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

    def run(self):
        """
        Perform the conjugate gradient experiment.
        """
        self._err = []

        x_k = self._x0
        r_k = self._A @ x_k - self._b  # type: np.ndarray
        p_k = -1 * r_k
        k = 0

        while k < self.n and np.linalg.norm(r_k) < self._tol:
            alpha_k = np.dot(r_k, r_k) / (p_k.transpose() @ self._A @ p_k)
            r_k_1 = r_k + alpha_k * self._A @ p_k

            beta_k_1 = np.dot(r_k_1, r_k_1) / np.dot(r_k, r_k)

            # Increment the counter and vectors
            k = k + 1
            x_k = x_k + alpha_k *p_k
            p_k = -r_k_1 + beta_k_1 * p_k
            r_k = r_k_1
            self.update_error(x_k)

    def update_error(self, x_k: np.ndarray):
        """
        Calculate the error for x_k and update the error list.

        :param x_k:
        :return:
        """
        self._err.append(self._calculate_err(x_k) - self._err_opt)

    def _calculate_err(self, x: np.ndarray) -> float:
        """

        :param x: X position to be passed to the error function.
        :return:
        """
        err = 0.5 * (x.transpose() @ self._A @ x)  # "@" symbol in Python 3.5 is matrix multiply
        err -= self._b.transpose() @ x
        return err[0]

    def _find_x_opt(self):
        """
        Finds the optimal value that minimizes the convex function.
        """
        self._x_opt = np.linalg.solve(self._A, 2 * self._b)

        self._err_opt = self._calculate_err(self._x_opt)

    @staticmethod
    def _build_plot(x_data: List, y_data: List, log_x: bool=False, log_y: bool=False,
                    x_label: str="", y_label: str="", title: str =""):
        """
        Creates a MatPlotLib plot for the the test data.  It supports multiple sets of y_data
        (stored as a list of lists).  All must share the same x-data in this implementation.

        :param x_data:
        :param y_data: List of multiple Y values for the corresponding X data
        :param log_x: True if the X-axis should be logarithmic.
        :param log_y: True if the Y-axis should be logarithmic.
        :param x_label: Label to be used for the X-axis
        :param y_label: Label to be used for the Y-axis
        :param title: Graph title
        """
        fig = plt.figure()
        plt.plot(x_data, y_data, linestyle="-")

        plt.legend(loc="best")

        # Optionally plot on a logarithmic axis
        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('symlog')

        # Configure the label and title.
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if title:
            plt.title(title)
        # Ensure a consistent x-axis even if there is insufficient y-data.
        plt.xlim(min(x_data), max(x_data))
        plt.ylim(10 * min(y_data), 10 * max(y_data))

        # Save the figure with the plot
        # plt.show()
        filename = title.replace(" ", "_").replace(",", "").lower()
        fig.savefig(filename + ".png")
        plt.close(fig)


def run_cg(n, clusters, p=None):
    cg = ConjugateGradient()
    cg.n = n

    cg.initialize_x0()
    cg.initialize_b()

    if p is None:
        cg.initialize_matrix(clusters)
    else:
        cg.initialize_matrix(clusters, p)

    cg.run()


def main():
    run_cg(10, [[10, 1000]])
    run_cg(1000, [[9, 11], [999, 1001]], [.1, .9])


if __name__ == "__main__":
    main()
