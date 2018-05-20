import sys
import numpy as np


class LBFGS:
    STOP_ERR = 10 ** -14

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

        self.err = None

    def run(self):
        """
        Based on
        :return:
        """
        self._k = 0
        x_k = self._x0

        self.err = []
        self._S = []
        self._Y = []
        while True:
            self.err.append(self._f(x_k))
            if self.err[-1] < LBFGS.STOP_ERR:
                return

            # Update the stored memory parameters
            x_prev = x_k
            x_k = None
            self._S.append(x_k - x_prev)
            self._Y.append(self._g(x_k) - self._g(x_prev))
            self._k += 1

    def _f(self, x: np.ndarray) -> float:
        """
        Extended Rosenbrock function.

        :param x: Location parameter.

        :return: Value of the extended Rosenbrock function at point \p x.
        """
        f_x = 0
        for j in range(1, (self._n // 2) + 1):
            i = 2 * j - 1
            f_x += self._alpha * (x[i] - (x[i - 1] ** 2)) ** 2 + (1 - x[i - 1]) ** 2
        return f_x

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

    for _m in [1, 5, 10, 20, 100, sys.maxsize]:
        lbfgs.m = _m
        lbfgs.run()

        file_name = f"lbfgs_m={_m}.csv"
        with open(file_name, "w") as fout:
            fout.write("k,f_err")
            for k, err in enumerate(lbfgs.err):
                fout.write(f"\n{k},{err}")
