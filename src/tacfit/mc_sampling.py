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


def _log_prob_smpl(data: npt.NDArray[np.float64],
                   smpl: npt.NDArray[np.float64],
                   ln_sigma: float) -> float:

    # For each data point there are two contributions:
    #   1) the distance between the data and the sample relative to the
    #      uncertainty,
    #   2) the uncertainty itself

    n = data.size  # number of data points

    # Contribution 1:
    cont1 = -np.sum(((np.power(data - smpl, 2.0)) /
                     (2.0 * np.exp(2.0 * ln_sigma) * np.power(data, 2.0))))

    # Contribution 2:
    y2 = np.power(data, 2.0)
    lny2 = np.log(2.0 * np.pi * y2)
    cont2 = -n * ln_sigma - 0.5 * np.sum(lny2)

    return float(cont1 + cont2)


def _emcee_fcn(param_values: npt.NDArray[np.float64],
               param_names: list[str],
               model: Callable[[npt.NDArray[np.float64],
                                npt.NDArray[np.float64],
                                dict[str, float]],
                               npt.NDArray[np.float64]],
               time_data: npt.NDArray[np.float64],
               input_data: npt.NDArray[np.float64],
               tissue_data: npt.NDArray[np.float64],
               param_bounds: dict[str, tuple[float, float]]) -> float:
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

    # Resample input function given current uncertainty estimate
    resampled_input = _resample_gaussian(input_data,
                                         np.exp(params['__lnsigma']))

    # Calculate the model given the current parameters and the resampled input
    ymodel = model(time_data, resampled_input, **params)  # type: ignore

    # Calculate and return the log-proability distribution (non-normalised)
    log_prop_input = _log_prob_smpl(input_data,
                                    resampled_input,
                                    params['__lnsigma'])
    log_prop_tissue = _log_prob_smpl(tissue_data,
                                     ymodel,
                                     params['__lnsigma'])
    return log_prop_input + log_prop_tissue


def mc_sample(time_data: npt.NDArray[np.float64],
              tissue_data: npt.NDArray[np.float64],
              input_data: npt.NDArray[np.float64],
              labels: dict[str, str],
              model: Callable[[npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               dict[str, float]],
                              npt.NDArray[np.float64]],
              params: dict[str, dict[str, float]],
              nsteps: int,
              nwalkers: int,
              nworkers: int,
              burn: int,
              thin: int,
              tcut: Optional[int] = None,
              output: Optional[str] = None) -> None:

    # Sample the posterior parameter distribution space using Monte Carlo
    # simulations with the emcee-package

    # Input sanitation
    if tcut is None:
        tcut = time_data.size

    # Parameters need to be unpacked when passed to emcee
    param_start = []  # Contains the optimised parameters (from an actual fit)
    param_names = []  # Contains the name of the parameters in the model
    param_bounds = {}  # Contains the parameter bounds (priors)
    for param in params:
        param_names.append(param)
        param_start.append(params[param]['value'])
        param_bounds[param] = (params[param]['min'], params[param]['max'])

    # Cut data as required
    time_data_cut = time_data[0:tcut]
    input_data_cut = input_data[0:tcut]
    tissue_data_cut = tissue_data[0:tcut]

    # Dimensionality of the parameter space
    n_dim = len(param_start)

    # Start the walkers in a gaussian ball around the optimised parameters
    start_p = np.array(param_start) + 1e-5 * np.random.randn(nwalkers, n_dim)

    # Run as a multithreaded pool
    with Pool(nworkers) as pool:
        # Start MC
        sampler = emcee.EnsembleSampler(nwalkers, n_dim, _emcee_fcn,
                                        args=(param_names, model,
                                              time_data_cut, input_data_cut,
                                              tissue_data_cut, param_bounds),
                                        pool=pool)
        sampler.run_mcmc(start_p, nsteps, progress=False)

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
    ml50_fit = model(time_data[0:tcut],  # type: ignore
                     input_data[0:tcut],
                     **ml50)
    original_fit = model(time_data[0:tcut],  # type: ignore
                         input_data[0:tcut],
                         **original_values)

    ax.plot(time_data, tissue_data, 'gx', label=labels['tissue'])
    ax.plot(time_data, input_data, 'rx--', label=labels['input'])
    ax.plot(time_data[0:tcut], ml50_fit, 'k-', label="ML50 Fit")
    ax.plot(time_data[0:tcut], original_fit, 'k-', label="Original Fit")
    # Pick 100 random samples and plot the projection to illustrate
    # parameter variation
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        smpl_params = {}
        for i in range(len(param_names)):
            smpl_params[param_names[i]] = sample[i]
        smpl_model = model(time_data[0:tcut],  # type: ignore
                           input_data[0:tcut],
                           **smpl_params)
        ax.plot(time_data[0:tcut], smpl_model, "C1", alpha=0.1)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Mean ROI-activity concentration [Bq/mL]')

    plt.legend()
    plt.grid(visible=True)
    if output is None:
        plt.show()
    else:
        fit_png_path = os.path.join(output, "fit.png")
        plt.savefig(fit_png_path)
        plt.clf()
