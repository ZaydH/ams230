import argparse
import math
from typing import Callable, List
import matplotlib.pyplot as plt


class HW1:
    alpha0 = 0  # type: float

    f = None  # type: Callable

    opt = None  # type: List[float]
    min_err = None  # type: float

    gradient = None  # type: Callable

    build_phi = None  # type: Callable
    phi = None  # type: Callable
    d_phi = None  # type: Callable

    tolerance = 0.001

    max_line_search_attempts = 1000
    C1 = 0.1
    C2 = 0.9

    @ staticmethod
    def steepest_descent(c: float, x_0: List[float]):
        """
        Runs the steepest descent line search algorithm.

        :param c: Value for scalar in
        :param x_0: Initial location
        """
        x_k = [x_0]
        err = [abs(HW1.f(x_0) - HW1.min_err)]
        while err[-1] > HW1.tolerance:

            p = HW1.calculate_steepest_descent_direction(x_k[-1])
            HW1.build_phi(c, x_k[-1], p)
            alpha = HW1.line_search()

            # Update X and then calculate its error
            x_k.append(HW1.calculate_next_x(x_k[-1], alpha, p))
            err.append(HW1.calculate_err(x_k[-1]))

        log_err = [math.log10(e) for e in err]
        HW1._build_plot(List(range(0, len(log_err))), log_err, log_y=True, x_label="Iteration #",
                        y_label="Error", title="Steepest Descent Learning with c=%d" % c)

    @staticmethod
    def calculate_err(x: List[float]) -> float:
        """
        Calculates the error between the specified X and the minimum error.

        :param x: Input vector for function HW1.f
        :return: Error from the minimum error.
        """
        return abs(HW1.f(x) - HW1.min_err)

    @staticmethod
    def calculate_steepest_descent_direction(x_k: List[float]) -> List[float]:
        """
        Calculates and returns the steepest descent direction using the gradient of f.

        :param x_k: Current location
        :return: Steepest descent direction
        """
        return [-1 * p_i for p_i in HW1.gradient(x_k)]

    @staticmethod
    def calculate_next_x(x: List[float], alpha: float, p: List[float]) -> List[float]:
        """
        Calculates a new location such that:

        x_{k+1} = x_{k} + alpha * p_k.

        :param x: Current location
        :param alpha: Step length
        :param p: Descent direction

        :return: New x_{k+1}
        """
        assert len(x) == len(p)
        return [x[i] + alpha * p[i] for i in range(0, len(x))]

    @staticmethod
    def line_search() -> float:
        """
        Calculates alpha for line search.

        :return: alpha for the function.
        """
        prev_alpha = HW1.alpha0
        alpha = HW1.alpha0 + 1
        phi_alpha = HW1.phi(HW1.alpha0)  # This is overwritten inside the loop

        phi_al_0 = HW1.phi(HW1.alpha0)
        d_phi_al_0 = HW1.d_phi(HW1.alpha0)
        for i in range(1, HW1.max_line_search_attempts + 1):
            prev_phi_alpha = phi_alpha

            phi_alpha = HW1.phi(alpha)
            if phi_alpha > phi_al_0 + HW1.C1 * alpha * d_phi_al_0 \
                    or (i > 1 and phi_alpha > prev_phi_alpha):
                return HW1.zoom(prev_alpha, alpha)

            d_phi_alpha = HW1.d_phi(alpha)
            if abs(d_phi_alpha) <= -HW1.C2 * d_phi_al_0:
                return alpha
            if d_phi_alpha >= 0:
                return HW1.zoom(alpha, prev_alpha)

            prev_alpha = alpha
            alpha *= 2
        assert False

    @staticmethod
    def zoom(alpha_lo: float, alpha_hi: float) -> float:
        """
        Perform Binary search within the range above to find the best value.

        :param alpha_lo:
        :param alpha_hi:
        :return: Optimal alpha value for line search.
        """
        phi_al_0 = HW1.phi(HW1.alpha0)
        d_phi_al_0 = HW1.d_phi(HW1.alpha0)
        while True:
            alpha_m = (alpha_lo + alpha_hi) / 2
            phi_alpha = HW1.phi(alpha_m)

            if phi_alpha > phi_al_0 + HW1.C1 * alpha_m * d_phi_al_0 \
                    or phi_alpha >= HW1.phi(alpha_lo):
                alpha_hi = alpha_m
            else:
                d_phi_alpha = HW1.d_phi(alpha_m)
                if abs(d_phi_alpha) <= -HW1.C2 * d_phi_al_0:
                    return alpha_m
                if d_phi_alpha * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_m

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
            plt.yscale('log')

        # Configure the label and title.
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        if title:
            plt.title(title)
        # Ensure a consistent x-axis even if there is insufficient y-data.
        plt.xlim(0.9 * min(x_data), 1.1 * max(x_data))

        # Save the figure with the plot
        # plt.show()
        fig.savefig(title.replace(" ", "_").replace(",", "").lower() + ".png")
        plt.close(fig)

    @staticmethod
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


def build_phi(c: float, x_k: List[float], p_k: List[float]):
    """
    Calculates functions phi and phi' for use in line search.

    :param c: Parameter for function "f"
    :param x_k: Current location x_k
    :param p_k: Descent direction p_k
    """
    x_a_p = lambda i, alpha: x_k[i] + alpha * p_k[i]
    HW1.phi = lambda alpha: ((c * x_a_p(0, alpha) - 2) ** 4
                             + (x_a_p(1, alpha) ** 2) * ((c * x_a_p(0, alpha) - 2) ** 2)
                             + (x_a_p(1, alpha) + 1) ** 2)

    HW1.d_phi = lambda alpha: (4 * c * p_k[0] * (x_a_p(0, alpha) - 2) ** 3
                               + 2 * p_k[1] * (x_a_p(1, alpha) + 1)
                               + 2 * p_k[1] * x_a_p(1, alpha) * (c * x_a_p(0, alpha) - 2) ** 2
                               + (x_a_p(1, alpha) ** 2))


def build_p6_f_and_gradient(c):
    """
    Constructs the gradient function for problem #6 in HW1.
    :param c: Value for scalar "c" in function
    """
    HW1.f = lambda x: (c * x[0] - 2) ** 4 + (x[1] ** 2) * ((c * x[0] - 2) ** 2) + (x[1] + 1) ** 2

    HW1.opt = [2.0 / c, -1]
    HW1.min_err = HW1.f(HW1.opt)

    HW1.gradient = lambda x: [4 * c * ((c * x[0] - 2) ** 3) + 2 * c * (x[1] ** 2) * (c * x[0] - 2),
                              2 * x[1] * ((c * x[0] - 2) ** 2) + 2 * (x[1] + 1)]


def parse_args():
    """
    Simple parser for the command line arguments.

    :return: Parsed command line arguments
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('c', type=float, help='Function scalar')
    # arg_parser.add_argument('tol', type=float, help='Error tolerance for the learner')
    arg_parser.add_argument('x0_1', type=float, help='First coordinate of x_0')
    arg_parser.add_argument('x0_2', type=float, help='Second coordinate of x_0')

    return arg_parser.parse_args()


def main():
    HW1.setup_logger()
    args = parse_args()

    build_p6_f_and_gradient(args.c)
    HW1.build_phi = build_phi

    _x_0 = [args.x0_1, args.x0_2]
    HW1.steepest_descent(args.c, _x_0)


if __name__ == "__main__":
    main()
