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
                tcut: Optional[int | list[int]] = None,
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
                    the data points are included in the fit. If a list
                    is given, a the fit will be made for all values in the
                    list.
    delay       --  Delay the input function. This shifts the measurement times
                    of the input function, such that a point previously
                    measured at time t will be set to a new time t+delay. None
                    (default) means no delay (delay=0.0).
    output      --  If None, plots are shown on the screen.
                    If a path is given, plots are saved to files on that
                    path.
    """

    # Input sanitation:

    # If tcut is None set it to use all data points
    t_cut = [time_data.size]
    single_fit = True
    if isinstance(tcut, int):
        # An int was given as tcut: use as single value for t_cut
        t_cut = [tcut]
    elif isinstance(tcut, list):
        # A list was given: fit all values in the list
        single_fit = False
        t_cut = tcut


    # If no delay is set, we use the same time for input and tissue
    input_time = time_data.copy()
    if delay is not None:
        input_time = input_time + delay

    # Create container for results if running in multi_fit mode
    scan_res = {
        'tcuts': t_cut.copy(),
        'r2': np.zeros(len(t_cut))
    }
    for param in params:
        scan_res[param] = np.zeros((2, len(t_cut)))

    for i in range(len(t_cut)):

        # Create lmfit Parameters-object
        parameters = lmfit.create_params(**params)

        # Define model to fit
        fit_model = lmfit.Model(model,
                                independent_vars=['t_in', 'in_func', 't_out'])
        # Run fit from initial values
        res = fit_model.fit(tissue_data[0:t_cut[i]],
                            t_in=input_time[0:t_cut[i]],
                            in_func=input_data[0:t_cut[i]],
                            t_out=time_data[0:t_cut[i]],
                            params=parameters)

        if single_fit:
            # Report and plot result of fit

            lmfit.report_fit(res)
            # Calculate best fitting model
            best_fit = model(t_in=input_time[0:tc],  # type: ignore
                             in_func=input_data[0:t_cut[i]],
                             t_out=time_data[0:t_cut[i]],
                             **res.best_values)

            fig, ax = plt.subplots()
            ax.plot(time_data, tissue_data, 'gx', label=labels['tissue'])
            ax.plot(input_time, input_data, 'rx--', label=labels['input'])
            ax.plot(time_data[0:t_cut[i]], best_fit, 'k-', label="Fit")

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

        else:
            # Save results of fit before moving on to next tcut
            for param in params:
                scan_res[param][0][i] = res.params[param].value
                scan_res[param][1][i] = res.params[param].stderr
                scan_res['r2'][i] = res.rsquared

    if not single_fit:
        # Make a plot showing the fit scan
        n_params = len(params)

        # Make a colour map
        cmap = plt.cm.get_cmap('RdYlGn')
        norm = plt.Normalize(vmin=0.95, vmax=1.0)

        # Plot
        fig, ax = plt.subplots(n_params)
        i = 0
        for param in params:
            ax[i].set_title(param)
            ax[i].errorbar(
                scan_res['tcuts'],
                scan_res[param][0],
                yerr=scan_res[param][1],
                fmt='none',
                ecolor='gray',
                capsize=5
            )
            sc = ax[i].scatter(
                scan_res['tcuts'],
                scan_res[param][0],
                c=scan_res['r2'],
                cmap=cmap,
                norm=norm,
                marker='s',
                s=100
            )
            ax[i].grid(True)
            cbar = plt.colorbar(sc, ax=ax[i])
            cbar.set_label('r^2')

            i = i + 1
        plt.show()

