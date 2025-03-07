import lmfit
import matplotlib.pyplot as plt
from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
import os


def fit_leastsq(time_data: npt.NDArray[np.float64],
                tissue_data: npt.NDArray[np.float64],
                input_data: npt.NDArray[np.float64],
                model: Callable[[npt.NDArray[np.float64],
                                 npt.NDArray[np.float64],
                                 npt.NDArray[np.float64],
                                 dict[str, float]],
                                npt.NDArray[np.float64]],
                params: dict[str, dict[str, float]],
                labels: dict[str, str],
                tcut: Optional[int] = None,
                delay: Optional[float] = None,
                output: Optional[str] = None) -> None:
    """Fit a model to measured TAC data using lmfit.
    This minimised the sum of squared residuals using the least-squares method
    of the lmfit package. The residuals are defined as the distance between the
    measured tissue function and the modeled tissue function given the input
    function and the parameters of the model.

    Arguments:
    time_data   --  Array of time data
    tissue_data --  Array of measured tissue data
    input_data  --  Array of measured input function data
    model       --  The model to fit to the data
    params      --  Dict object setting initial values and bounds for the
                    parameters of the model. It must be structured like
                    {'param1': {'value': ...,
                                'min': ...,
                                'max': ...},
                     ...
                    }
    labels      --  Labels to use when plotting. Required keys are:
                    labels['input']:    Used in the input function data legend
                    labels['tissue']:   Used in the tissue data legend
    tcut        --  Fit only the first tcut data points. If None (default), all
                    the data points are included in the fit.
    delay       --  Delay the input function. This shifts the measurement times
                    of the input function, such that a point previously
                    measured at time t will be set to a new time t+delay. None
                    (default) means no delay (delay=0.0).
    output      --  If None, plots are shown on the screen.
                    If a path is given, plots are saved to files on that
                    path.
    """

    # Input sanitation:

    # If no tcut is set, set it equal to the number of points in the data
    tcut_sane = time_data.size
    if tcut is not None:
        tcut_sane = tcut

    # If no delay is set, we use the same time for input and tissue
    input_time = time_data.copy()
    if delay is not None:
        input_time = input_time + delay

    # Create lmfit Parameters-object
    parameters = lmfit.create_params(**params)

    # Define model to fit
    fit_model = lmfit.Model(model,
                            independent_vars=['t_in', 'in_func', 't_out'])
    # Run fit from initial values
    res = fit_model.fit(tissue_data[0:tcut_sane],
                        t_in=input_time[0:tcut_sane],
                        in_func=input_data[0:tcut_sane],
                        t_out=time_data[0:tcut_sane],
                        params=parameters)

    # Report!
    lmfit.report_fit(res)
    # Calculate best fitting model
    best_fit = model(t_in=input_time[0:tcut_sane],  # type: ignore
                     in_func=input_data[0:tcut_sane],
                     t_out=time_data[0:tcut_sane],
                     **res.best_values)

    fig, ax = plt.subplots()
    ax.plot(time_data, tissue_data, 'gx', label=labels['tissue'])
    ax.plot(input_time, input_data, 'rx--', label=labels['input'])
    ax.plot(time_data[0:tcut_sane], best_fit, 'k-', label="Fit")

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Mean ROI-activity concentration')

    plt.legend()
    plt.grid(visible=True)
    if output is None:
        plt.show()
    else:
        fit_png_path = os.path.join(output, "fit.png")
        plt.savefig(fit_png_path)
        plt.clf()
