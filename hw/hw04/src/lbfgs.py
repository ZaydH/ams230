import logging
import sys
import numpy as np


class LBFGS:
    STOP_ERR = 10 ** -14
    MAX_LINE_SEARCH_ATTEMPTS = 20

    def __init__(self, n: int, alpha: float):
        """
        Constructor for the Limited Memory BFGS function.

        :param n: Dimension of the function. (Must be even)
        :param alpha:
        """
        assert n > 0 and n % 2 == 0
        self._n = n
        self._alpha = alpha
        self.m = None

        self._x0 = -1 * np.ones(n)
        self._x_opt = np.ones(n)
        self._x = None

        self._H = None  # type: np.ndarray
        self._p = None

        # Values for line search
        self._alpha0 = 1
        self._c1 = 0.1
        self._c2 = 0.45

        self._k = None
        self._gamma = None
        self._S = None
        self._Y = None

        self.err = None

    def run(self):
        """
        Based on Algorithm 7.5 (L-BFGS) of Nocedal and Wright (pg. 179).
        """
        assert self.m is not None and self.m > 0

        self._k = 0
        self._x = self._x0

        self.err = []
        self._S = []
        self._Y = []
        while True:
            self.err.append(self._f(self._x))
            if self._k % 1 == 0:
                logging.info(f"m={self.m}, k={self._k} --- err=%.10f" % self.err[-1])

            if self.err[-1] < LBFGS.STOP_ERR:
                return

            # Limit the recursion depth
            if self._k > self.m:
                self._S = self._S[1:]
                self._Y = self._Y[1:]
            # Calculate gamma as defined in Eq. (7.20) of Nocedal and Wright
            if self._k == 0:
                # ToDo Confirm gamma_0 value
                self._gamma = 1
            else:
                self._gamma = (self._S[-1].T @ self._Y[-1]) / (self._Y[-1].T @ self._Y[-1])

            # ToDo Verify the value of d0
            # d_0 = np.zeros((self._n, self._n))
            d_0 = np.identity(self._n)

            self._H = self._bfgs_rec(d_0, min(self._k, self.m))
            self._p = -1 * self._H @ self._g(self._x)
            alpha_k = self._line_search()

            # Update the stored memory parameters
            x_prev = self._x
            self._x = self._x + alpha_k * self._p
            self._S.append(self._x - x_prev)
            self._Y.append(self._g(self._x) - self._g(x_prev))
            self._k += 1

    def _f(self, x: np.ndarray) -> float:
        """
        Extended Rosenbrock function.

        :param x: Location parameter.

        :return: Value of the extended Rosenbrock function at point \p x.
        """
        f_x = 0
        for _i in range(1, (self._n // 2) + 1):
            j = 2 * _i - 1
            f_x += self._alpha * (x[j] - (x[j - 1] ** 2)) ** 2 + (1 - x[j - 1]) ** 2
        return f_x

    def _phi(self, alpha: float) -> float:
        return self._f(self._x + alpha * self._p)

    def _d_phi(self, alpha):
        """
        Derivative of $\phi(x+\alpha p)$ for use in line search.

        :return:
        """
        return self._g(self._x + alpha * self._p).T @ self._p

    def _g(self, x) -> np.ndarray:
        """
        Gradient of the extended Rosenbrock function.

        :param x: Location x_j
        :return:
        """
        grad_odd = lambda _j: -4 * self._alpha * x[_j] * (x[_j + 1] - x[_j] ** 2) - 2 * (1 - x[_j])
        grad_even = lambda _j: 2 * self._alpha * (x[_j] - x[_j - 1] ** 2)

        grad = []
        for i in range(1, (self._n // 2) + 1):
            j = 2 * i - 1
            grad.append(grad_odd(j - 1))
            grad.append(grad_even(j))
        grad_np = np.array(grad)
        return grad_np

    def _bfgs_rec(self, d: np.ndarray, k: int) -> np.ndarray:
        """
        Recursive implementation of BFGS.

        :param d:
        :param k: Remaining depth of the recursion.
        :return:
        """
        if k == 0:
            return self._gamma * d
        else:
            # Subtract one since indexed from 0
            s_k = self._S[k - 1]
            y_k = self._Y[k - 1]

            alpha = (s_k.T @ d) / (y_k.T @ s_k)
            d = d - alpha * y_k
            d = self._bfgs_rec(d, k - 1)
            return d + (alpha - (y_k.T @ d) / (y_k.T @ s_k)) * s_k

    def _line_search(self) -> float:
        """
        Calculates alpha for line search.

        :return: alpha for the function.
        """
        prev_alpha = self._alpha0
        alpha = self._alpha0 + 1
        phi_alpha = self._phi(self._alpha0)  # This is overwritten inside the loop

        phi_al_0 = self._phi(self._alpha0)
        d_phi_al_0 = self._d_phi(self._alpha0)
        for i in range(1, LBFGS.MAX_LINE_SEARCH_ATTEMPTS + 1):
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
            if alpha_hi == alpha_lo:
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


def setup_logger(filename="tester.log", log_level=logging.DEBUG):
    """
    Logger Configurator

    Configures the test logger.

    :param filename: Log file name
    :type filename: str
    :param log_level: Level to log
    :type log_level: int
    """
    date_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM

    logging.basicConfig(filename=filename, level=log_level,
                        format='%(asctime)s -- %(levelname)s -- %(message)s',
                        datefmt=date_format)

    # Also print to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s -- %(levelname)s -- %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    logging.info("***************************** New Run Beginning ****************************")


if __name__ == "__main__":
    _n = int(sys.argv[1])
    _alpha = float(sys.argv[2])

    setup_logger()
    lbfgs = LBFGS(_n, _alpha)

    for _m in [sys.maxsize, 1, 5, 10, 20, 100, 1]:
        lbfgs.m = _m
        lbfgs.run()

        file_name = f"lbfgs_m={_m}.csv"
        with open(file_name, "w") as fout:
            fout.write("k,f_err")
            for _k, err in enumerate(lbfgs.err):
                fout.write(f"\n{_k},{err}")
