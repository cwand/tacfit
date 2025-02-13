import numpy as np
import numpy.typing as npt
from typing import Optional
import matplotlib.pyplot as plt


def load_table(path: str) -> dict[str, npt.NDArray[np.float64]]:
    """Loads a table saved with colibri.save_tac.

    Arguments:
    path    --  The filename of the file containing the table.

    Return value:
    A dict object with column headers as keys and data as values.
    """

    # Read labels from file
    with open(path) as f:
        header = f.readline()
    header_cols = header.split()
    header_cols = header_cols[1:]

    # Load data (excluding header)
    data = np.loadtxt(path)

    # Put data into a dict object with correct labels
    data_dict = {}
    for i in range(len(header_cols)):
        data_dict[header_cols[i]] = data[:, i]
    return data_dict


def plot_data_nofit(time_data: npt.NDArray[np.float64],
                    inp_data: npt.NDArray[np.float64],
                    tis_data: npt.NDArray[np.float64],
                    labels: Optional[dict[str, str]] = None,
                    out_file: Optional[str] = None):
    """Plots dynamic data without fits.

    Arguments:
    time_data:  --  The time data in a numpy array.
    inp_data:   --  Input function data in a numpy array. Will be plotted in
                    red colour.
    tis_data:   --  Tissue data in a numpy array. Will be plotted in black.
    labels:     --  Optional dict of labels to show in the legend and on the
                    time axis. To replace some or all of the labels, set:
                    labels['time'] = ...
                    labels['inp'] = ...
                    labels['tis'] = ...
    out_file:   --  If None the plot will be displayed on the screen.
                    If a path is given, the plot will be saved to a file.
    """

    fig, ax = plt.subplots()

    # Input sanitation
    if labels is None:
        labels = {}

    # Plot input function data
    label = 'input function'
    if 'inp' in labels:
        label = labels['inp']
    ax.plot(time_data, inp_data, 'ro--', label=label)

    # Plot tissue data
    label = 'tissue'
    if 'tis' in labels:
        label = labels['tis']
    ax.plot(time_data, tis_data, 'kx--', label=label)

    # Replace x axis label if required
    if 'time' in labels:
        ax.set_xlabel(labels['time'])
    else:
        ax.set_xlabel("Time")

    # Show legend, grid and figure
    plt.legend()
    plt.grid(visible=True)

    # Display or save figure
    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file)
        plt.clf()
