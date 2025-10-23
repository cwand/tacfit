import numpy as np
import numpy.typing as npt
from typing import Optional
import matplotlib.pyplot as plt


def load_table(path: str) -> dict[str, npt.NDArray[np.float64]]:

    """Loads a data table saved with numpy.savetxt.

    Arguments:
        path    --  The filename of the file containing the table.

    Returns:
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


def plot_tac(time_data: npt.NDArray[np.float64],
             input_data: npt.NDArray[np.float64],
             tissue_data: npt.NDArray[np.float64],
             labels: Optional[dict[str, str]] = None,
             output: Optional[str] = None):

    """Plot TAC-data.

    Arguments:
        time_data   --  Array of time data
        input_data  --  Array of input function data
        tissue_data --  Array of tissue data
        labels      --  Optional labels to use when plotting. Acknowledged keys
                        are:
                        labels['time']:     Used on the x-axis
                        labels['tacunit']:  Used on the y-axis
                        labels['input']:    Used in the input function data
                                            legend
                        labels['tissue']:   Used in the tissue data legend
        output      --  If None, the plot is shown on the screen.
                        If a path is given, the plot is saved to a file on that
                        path.

    """

    # Input sanitation
    if labels is None:
        labels = {}

    # Start plotting
    fig, ax = plt.subplots()

    label = "input"
    if 'input' in labels:
        label = labels['input']
    ax.plot(time_data, input_data, 'ro--', label=label)

    label = "tissue"
    if 'tissue' in labels:
        label = labels['tissue']
    ax.plot(time_data, tissue_data, 'kx--', label=label)

    if 'time' in labels:
        ax.set_xlabel(labels['time'])
    else:
        ax.set_xlabel("time")

    if 'tacunit' in labels:
        ax.set_ylabel(labels['tacunit'])

    # Show legend and grid
    plt.legend()
    plt.grid(visible=True)

    # Show or save figure
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
        plt.clf()


def create_corrected_input_function(orig_time_data: npt.NDArray[np.float64],
                                    orig_infunc_data: npt.NDArray[np.float64],
                                    delay: float = 0.0) \
        -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    corrected_time_data = np.array(orig_time_data) + delay
    corrected_infunc_data = np.array(orig_infunc_data)

    return corrected_time_data, corrected_infunc_data


def calc_weights(frame_duration: npt.NDArray[np.float64],
                 tissue_data: npt.NDArray[np.float64]) \
        -> npt.NDArray[np.float64]:
    return np.array(np.sqrt(np.abs(frame_duration / tissue_data)))
