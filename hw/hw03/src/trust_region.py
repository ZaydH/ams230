import argparse
import math
import numpy as np


class TrustRegion:

    UNIFORM_MIN = 10
    UNIFORM_MAX = 1000

    def __init__(self, n: int):
        """

        :param n: Dimension of positive definite, symmetric matrix Q
        """
        self._Q = TrustRegion.build_symmetric_pos_definite(n)
        self._x = None  # type: np.ndarray

    @staticmethod
    def build_symmetric_pos_definite(n: int) -> np.ndarray:
        """
        Constructs a symmetric, positive definite matrix with eigenvalues distributed uniformly
        between UNIFORM_MIN and UNIFORM_MAX.

        :param n: Dimension of the matrix.

        :return: Symmetric positive definite matrix with uniform eigenvalues.
        """
        eig = np.random.uniform(TrustRegion.UNIFORM_MIN, TrustRegion.UNIFORM_MAX, (n, 1))
        q, r = np.linalg.qr(np.random.rand(n, n))
        return q @ np.diag(eig) @ q.transpose()

    def f(self) -> float:
        """
        Value of the function

        f(x) = \log(1 + x Q x^{T})

        :return: Value of function f
        """
        quad_prod = self._x @ self._Q @ self._x.transpose()
        return math.log10(1 + quad_prod[0, 0])

    def g(self) -> np.ndarray:
        """
        Calculates the gradient of $f$
        :return: Vector for the gradient of $f$
        """
        return 2 * (self._Q @self._x) / (1 + self._x.transpose() @ self._Q @ self._x)


def parse_args() -> argparse.Namespace:
    """
    Parses the input arguments

    :return: Parsed input arguments
    """
    args = argparse.ArgumentParser()
    args.add_argument("n", "Dimension of matrix Q", type=int)
    return args.parse_args()

if __name__ == "__main__":
    _args = parse_args()
    tr = TrustRegion(_args.n)
