import sys
import numpy as np


class LBFGS:
    def __init__(self, n: int, alpha: float):
        """
        Constructor for the Limited Memory BFGS function.

        :param n: Dimension of the function
        :param alpha:
        """
        self._n = n
        self._alpha = alpha
        self.m = None

        self._x0 = -1 * np.ones(n)
        self._x_opt = np.ones(n)

        self._k = None
        self._S = None
        self._Y = None

        self._err = None

    def run(self):
        """
        Based on
        :return:
        """
        self._k = 0
        x_prev = self._x0

        self._err = [self._f(self._x0)]
        self._S = []
        self._Y = []
        while True:


            x_k = None

            # Update the stored memory parameters
            self._S.append(x_k - x_prev)
            self._Y.append(self._g(x_k) - self._g(x_prev))
            x_prev = x_k

    def _f(self, x):
        """
        Extended Rosenbrock function.

        :param x: Location parameter.

        :return: Value of the extended Rosenbrock function at point \p x.
        """
        f_x = 0
        for i in range(1, self._n // 2):
            f_x += self._alpha * (x[2 * i] - (x[2 * i - 1] ** 2)) ** 2 + (1 - x[2 * i - 1]) ** 2
        return f_x

    def _g(self, x):
        """

        :param x:
        :return:
        """
        

    def _bfgs_rec(self, d, gamma, k):
        """
        Recursive implementation of BFGS.

        :param d:
        :param gamma:
        :param k: Remaining depth of the recursion.
        :return:
        """
        if k == 0:
            return gamma * np.identity(self._n)
        else:
            s_k = self._S[k]
            y_k = self._Y[k]

            alpha = (s_k.T @ d) / (y_k.T @ s_k)
            d = d - alpha * y_k
            d = self._bfgs_rec(d, gamma, k - 1)
            return d + (alpha - (y_k.T @ d) / (y_k.T @ s_k)) * s_k


if __name__ == "__main__":
    _n = int(sys.argv[1])
    _alpha = float(sys.argv[2])

    lbfgs = LBFGS(_n, _alpha)


