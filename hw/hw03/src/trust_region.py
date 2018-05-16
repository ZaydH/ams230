import argparse
import math
import sys
from typing import List, Callable

import numpy as np


class TrustRegion:

    UNIFORM_MIN = 10
    UNIFORM_MAX = 1000

    DELTA_0 = 1
    DELTA_MAX = 10000

    EPSILON = 10 ** -8

    TOLERANCE = 10 ** -12

    def __init__(self, n: int):
        """

        :param n: Dimension of positive definite, symmetric matrix Q
        """
        self._n = n

        self._Q = TrustRegion.build_symmetric_pos_definite(n)
        self._x = None  # type: np.ndarray

        # Initialize to a random vector for {U(0,1)}^n
        self._x0 = np.random.uniform(0, 1, (n, 1))
        self._k = None

        self.calculate_p = None  # type: Callable

        self._delta = None
        self._eta = 0.1
        self._B = None

        self._p = None  # type: np.ndarray
        self._rho = None

    @staticmethod
    def build_symmetric_pos_definite(n: int) -> np.ndarray:
        """
        Constructs a symmetric, positive definite matrix with eigenvalues distributed uniformly
        between UNIFORM_MIN and UNIFORM_MAX.

        :param n: Dimension of the matrix.

        :return: Symmetric positive definite matrix with uniform eigenvalues.
        """
        q, _ = np.linalg.qr(np.random.rand(n, n))
        eig = []
        while len(eig) < n:
            eig.append(np.random.uniform(TrustRegion.UNIFORM_MIN, TrustRegion.UNIFORM_MAX))
        d = np.diag(eig)
        return q @ d @ q.T

    def run(self) -> List[float]:
        self._delta = TrustRegion.DELTA_0
        self._k = 0
        self._initialize_x()
        err = []
        while True:
            err.append(self.f(self._x))
            print("%d,%f" % (self._k, err[-1]))
            if err[-1] < TrustRegion.TOLERANCE:
                return err

            self._calculate_B()
            self._p = self.calculate_p(self._B, self.g(), self._delta)
            self._calculate_rho()

            # Update delta (optionally)
            if self._rho < 0.25:
                self._delta = 0.25 * self._delta
            else:
                if (self._rho > 0.75
                        and abs(np.linalg.norm(self._p) - self._delta) < TrustRegion.EPSILON):
                    self._delta = min(2 * self._delta, TrustRegion.DELTA_MAX)
                else:
                    pass

            # Update x (optionally)
            if self._rho > self._eta:
                self._x = self._x + self._p
            else:
                pass

            self._k += 1

    def f(self, x: np.ndarray) -> float:
        """
        Value of the function

        f(x) = \log(1 + x Q x^{T})

        :param x: Location x used to calculate the cost

        :return: Value of function f
        """
        quad_prod = x.T @ self._Q @ x
        return math.log10(1 + quad_prod)

    def g(self) -> np.ndarray:
        """
        Calculates the gradient of $f$

        :return: Vector for the gradient of $f$
        """
        return 2 * (self._Q @ self._x) / (1 + self._x.T @ self._Q @ self._x)

    def _initialize_x(self):
        """
        Initialize x to a random variable
        """
        self._x = np.random.rand(self._n)

    def m_k(self, p: np.ndarray) -> float:
        """
        Calculates m_k using the specified value of \p p.

        :param p: Descent direction.

        :return: Value of m_k given \p p and the other state variables
        """
        return self.f(self._x) + self.g().T @ p + 0.5 * p.T @ self._Q @ p

    # noinspection PyPep8Naming
    def _calculate_B(self):
        """
        Calculates the approximation of the Hessian B_k
        """
        self._B = self._Q

    def _calculate_rho(self):
        """
        Calculates $\rho$ based on equation (4.4) in Nocedal and Wright.  It then updates the
        $\rho$ parameter of the object.
        """
        rho = self.f(self._x) - self.f(self._x + self._p)
        rho /= self.m_k(np.zeros(self._n)) - self.m_k(self._p)
        self._rho = rho


# noinspection PyPep8Naming
def calculate_cauchy_points(B: np.ndarray, g: np.ndarray, delta: float) -> np.ndarray:
    """
    Calculates the descent direction via the Cauchy Points algorithm

    :param B: Approximation of the Hessian that is PD
    :param g: Gradient at x_k
    :param delta: Trust region distance

    :return: New direction
    """
    if g.T @ B @ g <= 0:
        tau = 1
    else:
        tau = min(1, np.linalg.norm(g) ** 3 / (delta * g.T @ B @ g))
    return -1 * tau * delta / np.linalg.norm(g) * g


# noinspection PyPep8Naming
def calculate_dog_leg(B: np.ndarray, g: np.ndarray, delta: float) -> np.ndarray:
    """
    Calculates the descent direction via the Dog Leg algorithm

    :param B: Approximation of the Hessian that is PD
    :param g: Gradient at x_k
    :param delta: Trust region distance

    :return: New direction
    """
    p_b = -1 * np.linalg.inv(B) @ g
    p_u = -1 * (g.T @ g) / (g.T @ B @ g) * g

    if np.linalg.norm(p_b) <= delta:
        return p_b

    if np.linalg.norm(p_u) >= delta:
        return -1 * delta * g / np.linalg.norm(g)

    for t in np.linspace(1, 2, num=100):
        if abs(np.linalg.norm(p_u + (1 - t)*(p_b - p_u)) ** 2 - delta ** 2) < TrustRegion.TOLERANCE:
            return p_u + (1 - t)*(p_b - p_u)


if __name__ == "__main__":
    tr = TrustRegion(int(sys.argv[1]))
    algs = [("cauchy", calculate_cauchy_points), ("dog_leg", calculate_dog_leg)]
    for (name, alg) in algs:
        tr.calculate_p = alg
        f_err = tr.run()
        with open(name + ".csv", "w") as fout:
            fout.write("x,f(x)")
            for i, err in enumerate(f_err):
                fout.write("\n%d,%.15f" % (i, err))
