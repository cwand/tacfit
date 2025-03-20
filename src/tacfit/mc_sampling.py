import numpy as np
import numpy.typing as npt
from typing import Callable, Optional
import emcee
import corner
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os


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
                                npt.NDArray[np.float64],
                                npt.NDArray[np.float64],
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

    # Calculate the model given the current parameters and the resampled input
    ymodel = model(input_time,
                   input_data,
                   time_data, **params)  # type: ignore

    # Calculate and return the log-proability distribution (non-normalised)
    if error_model == "const":
        return _log_prob_uconst(tissue_data,
                                ymodel,
                                params['sigma'])
    if error_model == "sqrt":
        return _log_prob_usqrt(tissue_data,
                               ymodel,
                               params['sigma'])
    else:
        exit("Exiting: Unknown error model")


def mc_sample(time_data: npt.NDArray[np.float64],
              tissue_data: npt.NDArray[np.float64],
              input_data: npt.NDArray[np.float64],
              labels: dict[str, str],
              model: Callable[[npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
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
              delay: Optional[float] = None,
              progress: bool = True,
              output: Optional[str] = None) -> None:

    # Sample the posterior parameter distribution space using Monte Carlo
    # simulations with the emcee-package

    # Input sanitation
    tcut_sane = time_data.size
    if tcut is not None:
        tcut_sane = tcut

    input_time = time_data.copy()
    if delay is not None:
        input_time = input_time + delay

    # Parameters need to be unpacked when passed to emcee
    param_start = []  # Contains the optimised parameters (from an actual fit)
    param_names = []  # Contains the name of the parameters in the model
    param_bounds = {}  # Contains the parameter bounds (priors)
    for param in params:
        param_names.append(param)
        param_start.append(params[param]['value'])
        param_bounds[param] = (params[param]['min'], params[param]['max'])

    # Cut data as required
    input_time_cut = input_time[0:tcut_sane]
    time_data_cut = time_data[0:tcut_sane]
    input_data_cut = input_data[0:tcut_sane]
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
                                              input_time_cut,
                                              time_data_cut, input_data_cut,
                                              tissue_data_cut, param_bounds,
                                              error_model),
                                        pool=pool)
        sampler.run_mcmc(start_p, nsteps, progress=progress)

    # Plot the history of each walker
    fig, axes = plt.subplots(n_dim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axvline(x=burn, linestyle="--", color="xkcd:azure")
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(param_names[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    if output is None:
        plt.show()
    else:
        samples_png_path = os.path.join(output, "samples.png")
        print("Saving samples image to file", samples_png_path, ".")
        plt.savefig(samples_png_path)
        plt.clf()

    # Plot acceptance fraction of each walker:
    plt.plot(sampler.acceptance_fraction, 'o')
    plt.xlabel('walker')
    plt.ylabel('acceptance fraction')
    if output is None:
        plt.show()
    else:
        samples_png_path = os.path.join(output, "acceptance.png")
        plt.savefig(samples_png_path)
        plt.clf()

    # Try to calculate autocorrelation times
    # (might fail if "steps" is too small)
    try:
        tau = sampler.get_autocorr_time(discard=burn, thin=thin)
        print("Autocorrelation times:")
        for i in range(len(param_names)):
            print(f'   {param_names[i]}: {tau[i]:.1f}')
    except Exception as err:
        print("Autocorrelation could not be estimated")
        print(err)

    # Make corner plot
    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    corner.corner(flat_samples, labels=param_names, truths=param_start)
    if output is None:
        plt.show()
    else:
        corner_png_path = os.path.join(output, "corner.png")
        plt.savefig(corner_png_path)
        plt.clf()

    # Print parameter quantiles and save 50% quantile as well as original
    # values for plotting
    print("Parameter quantiles (2.5%, 16%, 50%, 84%, 97.5%)")
    ml50 = {}
    original_values = {}
    for i in range(n_dim):
        mcmc = np.percentile(flat_samples[:, i], [2.5, 16, 50, 84, 97.5])
        print(param_names[i], ":", mcmc)
        ml50[param_names[i]] = mcmc[2]
        original_values[param_names[i]] = param_start[i]

    fig, ax = plt.subplots()
    ml50_fit = model(input_time_cut,  # type: ignore
                     input_data_cut,
                     time_data_cut,
                     **ml50)
    original_fit = model(input_time_cut,  # type: ignore
                         input_data_cut,
                         time_data_cut,
                         **original_values)

    # Pick 100 random samples and plot the projection to illustrate
    # parameter variation
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        smpl_params = {}
        for i in range(len(param_names)):
            smpl_params[param_names[i]] = sample[i]
        smpl_model = model(input_time_cut,  # type: ignore
                           input_data_cut,
                           time_data_cut,
                           **smpl_params)
        ax.plot(time_data_cut, smpl_model, "C1", alpha=0.1)

    ax.plot(time_data_cut, tissue_data_cut, 'gx', label=labels['tissue'])
    ax.plot(time_data_cut, input_data_cut, 'rx--', label=labels['input'])
    ax.plot(time_data_cut, ml50_fit, 'b-', label="ML50 Fit")
    ax.plot(time_data_cut, original_fit, 'k-', label="Original Fit")

    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Mean ROI-activity concentration [Bq/mL]')

    plt.legend()
    plt.grid(visible=True)
    if output is None:
        plt.show()
    else:
        fit_png_path = os.path.join(output, "fit_mc.png")
        plt.savefig(fit_png_path)
        plt.clf()
