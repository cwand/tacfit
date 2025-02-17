import argparse
import numpy as np
import tacfit
import sys
import importlib.metadata
import time
import os


def _create_params(param_str: list[list[str]]) -> dict[str, dict[str, float]]:
    # Create a parameter dict from the input argument list of strings
    res = {}

    for param in param_str:
        param_dict = {
            'value': float(param[1])
        }

        if param[2] == "x":
            param_dict['min'] = -np.inf
        else:
            param_dict['min'] = float(param[2])

        if param[3] == "x":
            param_dict['max'] = np.inf
        else:
            param_dict['max'] = float(param[3])

        res[param[0]] = param_dict

    return res


def main(sys_args: list[str]):

    # Get version number from pyproject.toml
    __version__ = importlib.metadata.version("tacfit")
    start_time = time.time_ns()

    print("Starting TACFIT", __version__)
    print()

    # Handle input
    parser = argparse.ArgumentParser()
    parser.add_argument("tac_path", help="Path to TAC-file")
    parser.add_argument("time_label", help="Label of time data")
    parser.add_argument("input_label", help="Label of input function data")
    parser.add_argument("tissue_label", help="Label of tissue data")
    parser.add_argument("model", help="Model to use for fitting. Use the "
                                      "option --list_models to see all "
                                      "available models.")
    parser.add_argument("--param", action='append', nargs=4,
                        metavar=("par", "ini", "min", "max"),
                        help="Set parameter initial guesses and bounds. "
                             "\"par\" is the name of the parameter as given "
                             "in --list_models. Use \"x\" to indicate no "
                             "bound. Each parameter in the model is set with "
                             "a separate --param.")
    parser.add_argument("--leastsq", action='store_true',
                        help="Fit the data to the chosen model using the "
                             "least squares method. Requires all model "
                             "parameters be set using --param.")
    parser.add_argument("--mcpost", nargs=5, type=int,
                        metavar=("steps", "walkers", "burn", "thin", "pool"),
                        help="Make a Monte Carlo sampling of the posterior "
                             "parameter probability distribution function.")
    parser.add_argument("--list_models", action='store_true',
                        help="List all available models and their "
                             "parameters.")
    parser.add_argument("--plot_nofit", action='store_true',
                        help="Show data plot without fitting")
    parser.add_argument("--save_figs", nargs=1, metavar="path",
                        help="Save figures to the given path rather than "
                             "showing them.")

    args = parser.parse_args(sys_args)

    # Load data file
    print(f'Loading data from {args.tac_path}.')
    tac = tacfit.load_table(args.tac_path)
    print()

    # Possible models to use for fitting:
    models = {
        "step2": {'func':  tacfit.model.model_step2,
                  'desc':  "Sum of two step functions "
                           "(amp1, extent1, amp2, extent2)."}
    }

    # List models option
    if args.list_models:
        print("List of available models (and parameters):")
        for model in models:
            print(f'-- {model}: {models[model]["desc"]}')
        print()

    # Get chosen model
    model_str = args.model
    print(f'Model: {model_str}')
    model_desc = models[model_str]['desc']
    print(f'Description: {model_desc}')
    print()

    # Get parameters:
    params = _create_params(args.param)
    print("Parameter settings:")
    for param in params:
        print(f'  {param}:')
        param_value = params[param]['value']
        param_min = params[param]['min']
        param_max = params[param]['max']
        print(f'     value: {param_value}')
        print(f'     min:   {param_min}')
        print(f'     max:   {param_max}')
    print()

    # Fit least squares if required
    if args.leastsq:
        print("Starting least squares fitting.")
        if args.save_figs is None:
            output = None
        else:
            output = args.save_figs[0]
        tacfit.fit_leastsq(tac[args.time_label],
                           tac[args.tissue_label],
                           tac[args.input_label],
                           models[model_str]['func'],  # type: ignore
                           params,
                           {'tissue': args.tissue_label,
                            'input': args.input_label},
                           output=output)
        print()

    # Monte Carlo sampling if required
    if 'mcpost' in args:
        print("Starting Monte Calor sampling.")
        mc_opts = args.mcpost
        print("Monte Carlo parameters:")
        print(f'  steps:   {mc_opts[0]}')
        print(f'  walkers: {mc_opts[1]}')
        print(f'  burn:    {mc_opts[2]}')
        print(f'  thin:    {mc_opts[3]}')
        print(f'  threads: {mc_opts[4]}')
        if args.save_figs is None:
            output = None
        else:
            output = args.save_figs[0]
        tacfit.mc_sample(tac[args.time_label],
                         tac[args.tissue_label],
                         tac[args.input_label],
                         {'tissue': args.tissue_label,
                          'input': args.input_label},
                         models[model_str]['func'],  # type: ignore
                         params,
                         mc_opts[0], mc_opts[1], mc_opts[4],
                         mc_opts[2], mc_opts[3],
                         output=output)
        print()

    # Plot data if required
    if args.plot_nofit:
        print("Plotting data without fit.")
        if args.save_figs is None:
            output = None
        else:
            output = os.path.join(args.save_figs[0], "plot_nogit.png")
        tacfit.plot_tac(tac[args.time_label],
                        tac[args.input_label],
                        tac[args.tissue_label],
                        labels={'time': "Time",
                                'input': args.input_label,
                                'tissue': args.tissue_label},
                        output=output)
        print()

    # Report successful end of program
    run_time = (time.time_ns() - start_time) * 1e-9
    print(f'TACFIT finished successfully in {run_time:.1f} seconds.')
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
