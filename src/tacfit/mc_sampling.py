import numpy as np
import numpy.typing as npt
from typing import Callable, Optional
import emcee
import corner
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import tacfit

import tacfit.model.integrate as integrate


def _resample_gaussian(input_data: npt.NDArray[np.float64],
                       sigma: float) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng()
    sigma_y = sigma * input_data
    return rng.normal(loc=input_data, scale=sigma_y)


def _log_prob_uconst(data: npt.NDArray[np.float64],
                     smpl: npt.NDArray[np.float64],
                     sigma: float) -> float:

    # Calculates the log-probability assuming sigma_i = sigma
    # (a constant uncertainty across all data points)

    # For each data point there are two contributions:
    #   1) the distance between the data and the sample relative to the
    #      uncertainty,
    #   2) the uncertainty itself

    n = data.size  # number of data points
    s2 = np.power(sigma, 2.0)

    # Contribution 1:
    cont1 = - np.sum(np.power(data - smpl, 2.0)) / (2.0 * s2)

    # Contribution 2:
    cont2 = -n * np.log(2.0 * np.pi * s2) / 2.0

    return float(cont1 + cont2)


def _log_prob_usqrt(data: npt.NDArray[np.float64],
                    smpl: npt.NDArray[np.float64],
                    sigma: float) -> float:

    # Calculates the log-probability assuming sigma_i^2 = sigma^2 * y_i
    # (uncertainty scales with square root of measured data)

    # Check whether any data has value 0.0, which would cause problems
    if np.any(data == 0.0):
        raise ZeroDivisionError("Measured data has 0.0-values "
                                "(zero uncertainty).")

    s2 = np.power(sigma, 2.0)

    t1 = np.log(2.0 * np.pi * s2 * data)
    t2 = np.power(data - smpl, 2.0) / (s2 * data)

    return float(-0.5 * np.sum(t1 + t2))


def _init_walkers(start_position: npt.NDArray[np.float64],
                  param_bounds: npt.NDArray[np.float64],
                  n_walkers: int) -> npt.NDArray[np.float64]:
    # Initialise a set of walkers. Each walker is assigned a uniformly
    # random position in the parameter space. The points will be centered
    # around the start_position, and no points will be closer to the parameter
    # bound that halfway from the start_position:
    #     |      ..x..  |
    #  |  .x. |
    #        |     .....x.....             |

    # Get the dimensionality of the parameter space
    n_dim = len(start_position)
    # Calculate distance between start position and parameter bounds
    x = np.abs(np.transpose(param_bounds) - start_position)
    # Find out whether we are closer to the minimum or maximum bound
    y = np.array(np.min(x, axis=0))
    # Pick random number in [-0.5, 0.5], scale with distance to the closest
    # bound and add to start position.
    return np.array(
        start_position + y*(np.random.rand(n_walkers, n_dim) - 0.5))


def _emcee_fcn(param_values: npt.NDArray[np.float64],
               param_names: list[str],
               model: Callable[[npt.NDArray[np.float64],
                                dict[str, float]],
                               npt.NDArray[np.float64]],
               input_time: npt.NDArray[np.float64],
               time_data: npt.NDArray[np.float64],
               input_data: npt.NDArray[np.float64],
               tissue_data: npt.NDArray[np.float64],
               param_bounds: dict[str, tuple[float, float]],
               error_model: str) -> float:
    # Defines the probability distribution used by emcee (below) to sample
    # the parameter distribution of a fit model

    # Put parameters into dict object (required by model interface)
    params = {}
    for value, name in zip(param_values, param_names):
        params[name] = value

    # Prior parameter distribution:
    # Assumed to be uniform with bounds. i.e., if a parameter is outside the
    # bounds, the likelihood is 0 (so the log-likelihood is -inf)
    for param in params:
        if not param_bounds[param][0] < params[param] < param_bounds[param][1]:
            return -np.inf

    # Calculate the model given the current parameters and the input
    ymodel = integrate.model(
        input_time,
        input_data,
        time_data,
        model,
        **params
    )

    # Calculate and return the log-proability distribution (non-normalised)
    if error_model == "const":
        return _log_prob_uconst(tissue_data,
                                ymodel,
                                params['_sigma'])
    if error_model == "sqrt":
        return _log_prob_usqrt(tissue_data,
                               ymodel,
                               params['_sigma'])
    else:
        exit("Exiting: Unknown error model")


def mc_sample(time_data: npt.NDArray[np.float64],
              tissue_data: npt.NDArray[np.float64],
              input_data: npt.NDArray[np.float64],
              labels: dict[str, str],
              model: Callable[[npt.NDArray[np.float64],
                               dict[str, float]],
                              npt.NDArray[np.float64]],
              params: dict[str, dict[str, float]],
              error_model: str,
              nsteps: int,
              nwalkers: int,
              nworkers: int,
              burn: int,
              thin: int,
              tcut: Optional[int] = None,
              scut: Optional[float] = None,
              delay: Optional[float] = None,
              progress: bool = True,
              output: Optional[str] = None) -> None:

    # Sample the posterior parameter distribution space using Monte Carlo
    # simulations with the emcee-package

    # Input sanitation
    tcut_sane = time_data.size
    if tcut is not None:
        tcut_sane = tcut
    elif scut is not None:
        tcut_sane = int(np.searchsorted(time_data, scut))

    # If delay is None, use the value 0.0 (no delay)
    t_d = 0.0
    if delay is not None:
        t_d = delay
    # Make new input function from this delay:
    corr_input_time, corr_input_data = (
        tacfit.create_corrected_input_function(time_data, input_data, t_d))

    # Parameters need to be unpacked when passed to emcee
    param_start = []  # Contains the optimised parameters (from an actual fit)
    param_names = []  # Contains the name of the parameters in the model
    param_bounds = {}  # Contains the parameter bounds (priors)
    for param in params:
        param_names.append(param)
        param_start.append(params[param]['value'])
        param_bounds[param] = (params[param]['min'], params[param]['max'])

    # Cut data as required
    # input_time_cut = input_time[0:tcut_sane]
    time_data_cut = time_data[0:tcut_sane]
    # input_data_cut = input_data[0:tcut_sane]
    tissue_data_cut = tissue_data[0:tcut_sane]

    # Dimensionality of the parameter space
    n_dim = len(param_start)

    # Initialise walkers
    # First reformat parameter bounds to a 2d array
    pb = np.zeros((n_dim, 2))
    for i in range(n_dim):
        pb[i, 0] = param_bounds[param_names[i]][0]
        pb[i, 1] = param_bounds[param_names[i]][1]
    start_p = _init_walkers(np.array(param_start), pb, nwalkers)

    # Run as a multithreaded pool
    with Pool(nworkers) as pool:
        # Start MC
        sampler = emcee.EnsembleSampler(nwalkers, n_dim, _emcee_fcn,
                                        args=(param_names, model,
                                              corr_input_time,
                                              time_data_cut, corr_input_data,
                                              tissue_data_cut, param_bounds,
                                              error_model),
                                        pool=pool)
        sampler.run_mcmc(start_p, nsteps, progress=progress)

    # Get chain (walker specific and flat)
    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)

    # Plot the history of each walker
    fig, axes = plt.subplots(n_dim, figsize=(10, 7), sharex=True)
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axvline(x=burn, linestyle="--", color="xkcd:azure")
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(param_names[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    if output is not None:
        samples_png_path = os.path.join(output, "samples.png")
        print("Saving samples image to file", samples_png_path, ".")
        plt.savefig(samples_png_path)
        plt.clf()

    # Plot acceptance fraction of each walker:
    plt.plot(sampler.acceptance_fraction, 'o')
    plt.xlabel('walker')
    plt.ylabel('acceptance fraction')
    if output is not None:
        samples_png_path = os.path.join(output, "acceptance.png")
        plt.savefig(samples_png_path)
        plt.clf()

    # Make corner plot
    corner.corner(flat_samples, labels=param_names, truths=param_start)
    if output is not None:
        corner_png_path = os.path.join(output, "corner.png")
        plt.savefig(corner_png_path)
        plt.clf()

    # Calculate parameter statistics

    # Autocorrelation
    # (might fail if "steps" is too small)
    try:
        tau = sampler.get_autocorr_time(discard=burn, thin=thin)
    except Exception as err:
        tau = np.nan * np.ones_like(param_start)
        print("Autocorrelation could not be estimated")
        print(err)

    # Mean and uncertainty
    n = flat_samples.shape[0]
    means = np.mean(flat_samples, axis=0)
    std = np.std(flat_samples, axis=0, ddof=1)

    # Sample percentiles
    pct = np.percentile(flat_samples, [2.5, 16, 50, 84, 97.5], axis=0)

    # Report parameter values
    print(f'Parameter statistics ({n} samples):')
    print("Parameter\tMean\tStd.\t"
          "P2.5\tP16\tP50\tP84\tP97.5\tInt. autocorrelation time")
    for i in range(len(param_names)):
        print(f'{param_names[i]}\t{means[i]}\t{std[i]}\t'
              f'{pct[0][i]}\t{pct[1][i]}\t{pct[2][i]}\t'
              f'{pct[3][i]}\t{pct[4][i]}\t{tau[i]}')

    # Plotting

    # Find mean fit and fit using the starting position
    mean_values = {}
    original_values = {}
    for i in range(n_dim):
        mean_values[param_names[i]] = means[i]
        original_values[param_names[i]] = param_start[i]

    # Pick 100 random samples and plot the projection to illustrate
    # parameter variation
    inds = np.random.randint(len(flat_samples), size=100)

    # Plot IRF estimates
    fig, ax = plt.subplots()
    tt: npt.NDArray[np.float64] = np.arange(0.0, time_data[tcut_sane],
                                            0.01)
    mean_irf = model(tt, **mean_values)  # type: ignore
    original_irf = model(tt, **original_values)  # type: ignore
    for ind in inds:
        sample = flat_samples[ind]
        smpl_params = {}
        for i in range(len(param_names)):
            smpl_params[param_names[i]] = sample[i]
        smpl_model = model(tt, **smpl_params)  # type: ignore
        ax.plot(tt, smpl_model, "C1", alpha=0.1)
    ax.plot(tt, mean_irf, 'b-', label="Mean IRF")
    ax.plot(tt, original_irf, 'k-', label="Original IRF")
    plt.legend()
    plt.grid(visible=True)
    if output is not None:
        fit_png_path = os.path.join(output, "irf_mc.png")
        plt.savefig(fit_png_path)
        plt.clf()

    # Plot FIT
    fig, ax = plt.subplots()
    mean_fit = integrate.model(corr_input_time,  # type: ignore
                               corr_input_data,
                               time_data_cut,
                               model,
                               **mean_values)
    original_fit = integrate.model(corr_input_time,  # type: ignore
                                   corr_input_data,
                                   time_data_cut,
                                   model,
                                   **original_values)

    for ind in inds:
        sample = flat_samples[ind]
        smpl_params = {}
        for i in range(len(param_names)):
            smpl_params[param_names[i]] = sample[i]
        smpl_model = integrate.model(corr_input_time,  # type: ignore
                                     corr_input_data,
                                     time_data_cut,
                                     model,
                                     **smpl_params)
        ax.plot(time_data_cut, smpl_model, "C1", alpha=0.1)

    ax.plot(time_data_cut, tissue_data_cut, 'gx', label=labels['tissue'])
    input_time_plot = corr_input_time[0:tcut_sane]
    if '_delay' in mean_values:
        input_time_plot += mean_values['_delay']
    ax.plot(input_time_plot, input_data[0:tcut_sane],
            'rx--', label=labels['input'])
    ax.plot(time_data_cut, mean_fit, 'b-', label="Mean Fit")
    ax.plot(time_data_cut, original_fit, 'k-', label="Original Fit")

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Mean ROI-activity concentration [Bq/mL]')

    plt.legend()
    plt.grid(visible=True)
    if output is not None:
        fit_png_path = os.path.join(output, "fit_mc.png")
        plt.savefig(fit_png_path)
        plt.clf()
    else:
        plt.show()
