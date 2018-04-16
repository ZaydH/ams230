from typing import Callable
import matplotlib.pyplot as plt


class HW1:
    alpha0 = 0  # type: float
    phi = None  # type: Callable
    d_phi = None  # type: Callable
    tolerance = 0.001

    max_line_search_attempts = 1000
    C1 = 0
    C2 = 0

    def steepest_descent(self, x_0):
        X = [x_0]
        while



        HW1._build_plot()



    @staticmethod
    def line_search() -> float:
        """

        :return:
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
            phi_al_lo = HW1.phi(alpha_lo)

            if phi_alpha > phi_al_0 + HW1.C1 * alpha_m * d_phi_al_0 or phi_alpha >= phi_al_lo:
                alpha_hi = alpha_m
            else:
                d_phi_alpha = HW1.d_phi(alpha_m)
                if abs(d_phi_alpha) > -HW1.C2 * d_phi_al_0:
                    return alpha_m
                if d_phi_alpha * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_m

    @staticmethod
    def _build_plot(x_data, y_data, log_x=False, log_y=False,
                    x_label="", y_label="", title="", color=None, marker=None):
        """
        Creates a MatPlotLib plot for the the test data.  It supports multiple sets of y_data (stored
        as a list of lists).  All must share the same x-data in this implementation.

        :param x_data:
        :type x_data: List[float]
        :param y_data: List of multiple Y values for the corresponding X data
        :type y_data: List[List[float]]
        :param log_x: True if the X-axis should be logarithmic.
        :type log_x: bool
        :param log_y: True if the Y-axis should be logarithmic.
        :type log_y: bool
        :param x_label: Label to be used for the X-axis
        :type x_label: str
        :param y_label: Label to be used for the Y-axis
        :type y_label: str
        :param title: Graph title
        :type title: str
        """
        fig = plt.figure()
        plt.plot(x_data, y_data, linestyle="-", marker=marker, color=color)

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

def build_phi(c, x_k, p_k):
    HW1.phi = lambda alpha : c ((x_k[0] * alpha * p_k[0]) ** 4) \
                             + ((x_k[1] + alpha * p_k[1]) ** 2) * (c * (x_k[0] - 2) ** 2) \
                             + ((x_k[1] + alpha * p_k[1]) + 1) ** 2



