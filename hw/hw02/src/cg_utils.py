import logging
import sys

import matplotlib.pyplot as plt
from typing import List


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


def build_plot(x_data: List, y_data: List, log_x: bool=False, log_y: bool =False,
               x_label: str ="", y_label: str ="", title: str =""):
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

    # plt.legend(loc="best")

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
    plt.ylim(-5 + min(y_data), 5 + max(y_data))

    # Save the figure with the plot
    # plt.show()
    filename = title.replace(" ", "_").replace(",", "").lower()
    fig.savefig(filename + ".png")
    plt.close(fig)

    # Export a CSV of the data for plotting in LaTeX
    with open(filename + ".csv", "w") as fout:
        fout.write("%s,%s" % (x_label, y_label))
        for x_i, y_i in zip(x_data, y_data):
            fout.write("\n%.16f,%0.16f" % (x_i, y_i))
