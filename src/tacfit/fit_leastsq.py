import lmfit
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union

import numpy
import numpy as np
import numpy.typing as npt
import os

from tqdm import tqdm

import tacfit
import tacfit.model.integrate as integrate


def _print_fit(result: lmfit.model.ModelResult,
               delay: float,
               tcut: int,
               path: str):
    with open(path, "w") as f:
        f.write(f'r^2\t{result.rsquared}\n')
        f.write(f'delay\t{delay}\n')
        f.write(f'tcut\t{tcut}\n')
        for param in result.params:
            f.write(f'{param}\t{result.params[param].value}\t'
                    f'{result.params[param].stderr}\n')


def fit_leastsq(time_data: npt.NDArray[np.float64],
                tissue_data: npt.NDArray[np.float64],
                input_data: npt.NDArray[np.float64],
                model: Callable[[npt.NDArray[np.float64],
                                 dict[str, float]],
                                npt.NDArray[np.float64]],
                params: dict[str, dict[str, float]],
                labels: dict[str, str],
                tcut: Optional[Union[int, list[int]]] = None,
                scut: Optional[float] = None,
                delay: Optional[float] = None,
                confint: bool = True,
                output: Optional[str] = None,
                progress: bool = True) -> None:
    """Fit a model to measured TAC data using lmfit.
    This minimised the sum of squared residuals using the least-squares method
    of the lmfit package. The residuals are defined as the distance between the
    measured tissue function and the modeled tissue function given the input
    function and the parameters of the model.

    Arguments:
    time_data   --  Array of time data
    tissue_data --  Array of measured tissue data
    input_data  --  Array of measured input function data
    model       --  The IRF-model to fit to the data
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
                    is given, a fit will be made for all values in the
                    list.
    scut        --  Fit only data up to time scut. If None (default), all data
                    points are included in the fit. If both tcut and scut is
                    set, only the tcut value is used.
    delay       --  Delay the input function. This shifts the measurement times
                    of the input function, such that a point previously
                    measured at time t will be set to a new time t+delay. None
                    (default) means no delay (delay=0.0).
    output      --  If None, plots are shown on the screen.
                    If a path is given, plots are saved to files on that
                    path.
    progress    --  Whether to show a progress bar if doing a scan
    """

    # If delay is None, use the value 0.0 (no delay)
    t_d = 0.0
    if delay is not None:
        t_d = delay
    # Make new input function from this delay:
    corr_input_time, corr_input_data = (
        tacfit.create_corrected_input_function(time_data, input_data, t_d))

    if tcut is None:
        # Check if scut is set
        if scut is not None:
            # Find the largest time point larger than scut
            tcut = int(np.searchsorted(time_data, scut))
        else:
            # Use all data if not cut-off is set
            tcut = len(time_data)

    if isinstance(tcut, int):

        # Create lmfit Parameters-object
        parameters = lmfit.create_params(**params)

        # Define model to fit
        fit_model = lmfit.Model(integrate.model,
                                independent_vars=['t_in', 'in_func',
                                                  't_out', 'irf'])

        # Run fit from initial values
        res = fit_model.fit(tissue_data[0:tcut],
                            t_in=corr_input_time,
                            in_func=corr_input_data,
                            t_out=time_data[0:tcut],
                            irf=model,
                            params=parameters)

        # Report and plot result of fit

        lmfit.report_fit(res)

        if confint:
            print("[[Confidence Intervals]]")
            ci_report = res.ci_report()
            print(ci_report)

        # Show best fitting IRF:
        fig, ax = plt.subplots()
        tt: npt.NDArray[np.float64] = np.arange(0.0, time_data[tcut],
                                                0.01)
        best_irf = model(tt, **res.best_values)  # type: ignore
        ax.plot(tt, best_irf, 'k-', label="Fitted IRF")

        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('IRF')

        plt.legend()
        plt.grid(visible=True)
        if output is not None:
            # Save figure to file
            fit_png_path = os.path.join(output, "irf.png")
            plt.savefig(fit_png_path)
            plt.clf()

        # Calculate best fitting model
        best_fit = integrate.model(corr_input_time,
                                   corr_input_data,
                                   time_data[0:tcut],
                                   model,
                                   **res.best_values)

        # Plot results
        if '_delay' in params:
            corr_input_time = corr_input_time + res.best_values['_delay']
        fig, ax = plt.subplots()
        ax.plot(time_data[0:tcut], tissue_data[0:tcut],
                'gx', label=labels['tissue'])
        ax.plot(corr_input_time[0:tcut], corr_input_data[0:tcut],
                'rx--', label=labels['input'])
        ax.plot(time_data[0:tcut], best_fit, 'k-', label="Fit")

        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Mean ROI-activity concentration')

        plt.legend()
        plt.grid(visible=True)
        if output is not None:
            # Save figure to file
            _print_fit(res,
                       t_d,
                       tcut,
                       os.path.join(output, "result.txt"))
            fit_png_path = os.path.join(output, "fit.png")
            plt.savefig(fit_png_path)
            plt.clf()

    else:

        # Make arrays for multi-fit mode
        param_idx = {}
        idx = 0
        for param in params:
            param_idx[param] = idx
            idx = idx + 1
        r2s = np.zeros(len(tcut))
        scan_res = np.zeros((2 * len(params), len(tcut)))

        for i in tqdm(range(len(tcut)), disable=(not progress)):
            # Iterate over tcuts

            # Create lmfit Parameters-object
            parameters = lmfit.create_params(**params)

            # Define model to fit
            fit_model = lmfit.Model(integrate.model,
                                    independent_vars=['t_in', 'in_func',
                                                      't_out', 'irf'])

            # Run fit from initial values
            res = fit_model.fit(tissue_data[0:tcut[i]],
                                t_in=corr_input_time,
                                in_func=corr_input_data,
                                t_out=time_data[0:tcut[i]],
                                irf=model,
                                params=parameters)

            # Save results of fit before moving on to next tcut
            for param in params:
                scan_res[2*param_idx[param]][i] = res.params[param].value
                scan_res[2*param_idx[param] + 1][i] = res.params[param].stderr
            r2s[i] = res.rsquared

        # Make a plot showing the fit scan
        n_params = len(params)

        # Plot
        fig, axs = plt.subplots(nrows=(n_params+1))
        i = 0
        for param in params:
            axs[i].set_ylabel(param)
            axs[i].errorbar(
                 tcut,
                 scan_res[2*param_idx[param]],
                 yerr=scan_res[2*param_idx[param] + 1],
                 fmt='s',
                 capsize=4
            )
            axs[i].grid(True)

            i = i + 1

        axs[i].set_ylabel('r^2')
        axs[i].set_xlabel('tcut')
        axs[i].scatter(
             tcut,
             r2s,
             marker='x'
        )
        axs[i].set_ylim(bottom=0.95, top=1.0)
        axs[i].set_yticks(np.arange(0.95, 1.0, 0.01))
        axs[i].set_yticks(np.arange(0.95, 1.0, 0.002), minor=True)
        axs[i].grid(which='major', alpha=0.5)
        axs[i].grid(axis='y', which='minor', alpha=0.2)

        if output is not None:
            fit_png_path = os.path.join(output, "scan.png")
            plt.savefig(fit_png_path)
            plt.clf()

    if output is None:
        # Show all figures
        plt.show()
