import logging
import math
import sys
from enum import Enum
import numpy as np
from typing import Callable, List

from cg_utils import build_plot, setup_logger


class BetaMethod(Enum):
    """
    Enumerated class to
    """
    FR = "Fletcher-Reeves"
    FR_with_Restart = "Fletcher-Reeves with Restart"
    PR = "Polak-Ribiere"


class NonLinearConjugate:
    MAX_LINE_SEARCH_ATTEMPTS = 20

    def __init__(self, n: int):
        self.n = n
        logging.debug("Creating Non-Linear Conjugate Gradient with n=%d" % self.n)

        self._f = None
        # self._gradient = None

        self.calc_beta = None  # type: Callable

        self._x0 = None  # type: np.ndarray
        self._x_k = None  # type: np.ndarray
        self._p_k = None  # type: np.ndarray
        self._x_opt = None  # type: np.ndarray

        self.err = None  # type: List[float]
        self._err_opt = None  # type: float

        self._TOL = 5 * 10 ** -8

        self._alpha0 = 0
        self._c1 = 0.1
        self._c2 = 0.9

    def run(self, beta_method):
        """
        Perform non-linear conjugate gradient using the specified methods for calculating
        $\beta$.
        """
        logging.debug("Non-Linear Conjugate Gradient Beta Method -- %s" % beta_method.value)
        if beta_method == BetaMethod.FR:
            self.calc_beta = NonLinearConjugate.fr
        elif beta_method == BetaMethod.FR_with_Restart:
            self.calc_beta = NonLinearConjugate.fr_with_restart
        elif beta_method == BetaMethod.PR:
            self.calc_beta = NonLinearConjugate.pr
        else:
            raise ValueError("Unknown method for calculating beta")

        self.err = []
        self._x_k = self._x0
        self.update_err()
        grad_k = self._gradient(self._x_k)
        self._p_k = -grad_k
        k = 0

        # while k < self.n and np.linalg.norm(grad_x, 2) > self._TOL:
        while k < self.n ** 2 and np.linalg.norm(grad_k, 2) > self._TOL:
            alpha_k = self.line_search()
            x_k_1 = self._x_k + alpha_k * self._p_k

            grad_k_1 = self._gradient(x_k_1)
            beta_k = self.calc_beta(grad_k_1, grad_k)

            # Update the terms for the next loop iteration
            self._x_k = x_k_1
            self.update_err()

            self._p_k = -grad_k_1 + beta_k * self._p_k
            grad_k = grad_k_1
            k += 1
            if k % 10 == 0:
                logging.debug("Non-Linear CG %s Running -- k = %d -- Err = %f"
                              % (beta_method.value, k, self.err[-1]))

        logging.info("Completed Non-Linear Conjugate Gradient with k=%d" % (k-1))

    def line_search(self) -> float:
        """
        Calculates alpha for line search.

        :return: alpha for the function.
        """
        prev_alpha = self._alpha0
        alpha = self._alpha0 + 1
        phi_alpha = self._phi(self._alpha0)  # This is overwritten inside the loop

        phi_al_0 = self._phi(self._alpha0)
        d_phi_al_0 = self._d_phi(self._alpha0)
        for i in range(1, NonLinearConjugate.MAX_LINE_SEARCH_ATTEMPTS + 1):
            prev_phi_alpha = phi_alpha

            phi_alpha = self._phi(alpha)
            if phi_alpha > phi_al_0 + self._c1 * alpha * d_phi_al_0 \
                    or (i > 1 and phi_alpha >= prev_phi_alpha):
                return self.zoom(prev_alpha, alpha)

            d_phi_alpha = self._d_phi(alpha)
            if abs(d_phi_alpha) <= -self._c2 * d_phi_al_0:
                return alpha
            if d_phi_alpha >= 0:
                return self.zoom(alpha, prev_alpha)

            prev_alpha = alpha
            alpha *= 2
        assert False

    def zoom(self, alpha_lo: float, alpha_hi: float) -> float:
        """
        Perform Binary search within the range above to find the best value.

        :param alpha_lo:
        :param alpha_hi:
        :return: Optimal alpha value for line search.
        """
        phi_al_0 = self._phi(self._alpha0)
        d_phi_al_0 = self._d_phi(self._alpha0)
        while True:
            alpha_m = (alpha_lo + alpha_hi) / 2
            # Handle the case where floating point error is too high
            if abs(alpha_hi - alpha_lo) < 10 ** -20:
                return alpha_m

            phi_alpha = self._phi(alpha_m)

            if phi_alpha > phi_al_0 + self._c1 * alpha_m * d_phi_al_0 \
                    or phi_alpha >= self._phi(alpha_lo):
                alpha_hi = alpha_m
            else:
                d_phi_alpha = self._d_phi(alpha_m)
                if abs(d_phi_alpha) <= -self._c2 * d_phi_al_0:
                    return alpha_m
                if d_phi_alpha * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_m

    def initialize_f(self):
        """
        Initialize the cost function f.
        """
        self._f = lambda x: sum([100 * ((x[_i] ** 2 - x[_i+1]) ** 2) + (x[_i] - 1) ** 2
                                 for _i in range(0, self.n - 1)])

    def _phi(self, alpha: float) -> float:
        """
        Cost function f(x + alpha * p).

        :param alpha: Scalar from inexact line search.
        :return: Calculated value for function phi.
        """
        return self._f(self._x_k + alpha * self._p_k)

    def _d_phi(self, alpha: float) -> float:
        """
        Calculate \phi' given the specified alpha.

        :param alpha: Exact line search alpha.
        :return:
        """
        x_al_p = self._x_k + alpha * self._p_k
        return sum([200 * (2 * self._p_k[i] * x_al_p[i] - self._p_k[i + 1])
                    * (x_al_p[i] ** 2 - x_al_p[i + 1])
                    + 2 * self._p_k[i] * (x_al_p[i] - 1)
                    for i in range(0, self.n - 1)])

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Construct the gradient for the specified x.

        :param x: Current position
        :return: Gradient vector
        """
        grad_i = lambda _i: 400 * x[_i] * (x[_i] ** 2 - x[_i + 1]) + 2 * (x[_i] - 1)
        grad_prev = lambda _i: -200 * (x[_i - 1] ** 2 - x[_i])

        grad = [grad_i(0)]
        for _i in range(1, self.n - 1):
            grad.append(grad_i(_i) + grad_prev(_i))
        grad.append(grad_prev(self.n - 1))

        grad_np = np.array(grad)
        return grad_np

    def initialize_x0(self):
        """
        Initialize x0 to a random list as specified in the homework.
        """
        self._x0 = 2 * np.random.rand(self.n, 1)

    def initialize_x_opt(self):
        """
        Initialize the optimal x value.
        :return:
        """
        self._x_opt = np.ones([self.n, 1])

        assert self._f is not None
        self._err_opt = self._f(self._x_opt)

    def update_err(self):
        """
        Calculate the error from ideal and append it to the list of errors.
        """
        diff = abs(self._f(self._x_k) - self._err_opt)
        if diff == 0:
            return
        self.err.append(math.log10(diff))

    @staticmethod
    def fr(grad_k_1: np.ndarray, grad_k: np.ndarray) -> float:
        """
        Calculate beta using the Fletcher-Reeves method.

        :param grad_k_1: Gradient value at x_{k+1}
        :param grad_k: Gradient value at x_{k}
        :return: Beta^{FR}_{k+1}
        """
        return (grad_k_1.transpose() @ grad_k_1) / (grad_k.transpose() @ grad_k)

    @staticmethod
    def fr_with_restart(grad_k_1: np.ndarray, grad_k: np.ndarray) -> float:
        """
        Calculate beta using the Fletcher-Reeves method.

        :param grad_k_1: Gradient value at x_{k+1}
        :param grad_k: Gradient value at x_{k}
        :return: Beta^{FR}_{k+1}
        """
        if np.abs(grad_k_1.transpose() @ grad_k) / np.linalg.norm(grad_k, 2) >= 0.1:
            return 0
        return NonLinearConjugate.fr(grad_k_1, grad_k)

    @staticmethod
    def pr(grad_k_1: np.ndarray, grad_k: np.ndarray) -> float:
        """
        Calculate beta using the Polak-Ribiere method.

        :param grad_k_1: Gradient value at x_{k+1}
        :param grad_k: Gradient value at x_{k}
        :return: Beta^{PR}_{k+1}
        """
        beta = (grad_k_1.transpose() @ (grad_k_1 - grad_k)) / (np.linalg.norm(grad_k, 2) ** 2)
        return beta


def main(n: int):
    """
    Perform the conjugate gradient test for different methods.

    :param n: Dimension of the matrix.
    """
    ncg = NonLinearConjugate(n)

    ncg.initialize_f()
    ncg.initialize_x0()
    ncg.initialize_x_opt()

    for beta_method in [BetaMethod.FR_with_Restart, BetaMethod.PR, BetaMethod.FR]:
        ncg.run(beta_method)
        build_plot(list(range(0, len(ncg.err))), ncg.err, x_label="Iteration",
                   y_label="Log Error", title=beta_method.value)


if __name__ == "__main__":
    setup_logger(log_level=logging.DEBUG)

    _n = int(sys.argv[1])
    main(_n)
