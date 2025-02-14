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
                                 dict[str, float]],
                                npt.NDArray[np.float64]],
                params: dict[str, dict[str, float]],
                labels: dict[str, str],
                tcut: int,
                output: Optional[str]) -> None:
    # Fit a TAC to a given function using lmfit

    # Create lmfit Parameters-object
    parameters = lmfit.create_params(**params)

    # Define model to fit
    fit_model = lmfit.Model(model, independent_vars=['t', 'in_func'])
    # Run fit from initial values
    res = fit_model.fit(tissue_data[0:tcut],
                        t=time_data[0:tcut],
                        in_func=input_data[0:tcut],
                        params=parameters)

    # Report!
    lmfit.report_fit(res)
    # Calculate best fitting model
    best_fit = model(time_data[0:tcut],
                     input_data[0:tcut],
                     res.best_values)

    fig, ax = plt.subplots()
    ax.plot(time_data, tissue_data, 'gx', label=labels['tissue'])
    ax.plot(time_data, input_data, 'rx--', label=labels['input'])
    ax.plot(time_data[0:tcut], best_fit, 'k-', label="Fit")

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
