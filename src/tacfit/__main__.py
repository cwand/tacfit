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

        # Handle optional bounds: x means no bound in that direction
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
    parser.add_argument("--weight", metavar="FRAME_DUR",
                        help="Weight the residuals by the number of counts"
                             "in each frame. FRAME_DUR is the label of the"
                             "frame durations in the TAC-file.")
    tcut_group = parser.add_mutually_exclusive_group()
    tcut_group.add_argument("--tcut", metavar="N", type=int, nargs='*',
                            help="Cut the data at the N\'th data point. It is "
                            "possible to provide a list of values to scan"
                            "through them.")
    tcut_group.add_argument("--scut", metavar="T", type=float,
                            help="Cut the data after T seconds (time units).")
    parser.add_argument("--delay", type=float, metavar="T",
                        help="Delay the input function signal for T seconds.")
    parser.add_argument("--param", action='append', nargs=4,
                        metavar=("PAR", "INI", "MIN", "MAX"),
                        help="Set parameter initial guesses and bounds. "
                             "\"PAR\" is the name of the parameter as given "
                             "in --list_models. Use \"x\" to indicate no "
                             "bound. Each parameter in the model is set with "
                             "a separate --param.")
    parser.add_argument("--exprparam", action='append', nargs=2,
                        metavar=("PAR", "EXPR"),
                        help="Set a model parameter to a fixed exression (i.e."
                             " not fitted), which may involve other model "
                             "parameters.")
    parser.add_argument("--leastsq", action='store_true',
                        help="Fit the data to the chosen model using the "
                             "least squares method. Requires all model "
                             "parameters be set using --param.")
    parser.add_argument("--no_confint", action='store_false',
                        help="Skips computation of confidence intervals when "
                             "using leastsq, which can be an expensive "
                             "computation in some models.")
    parser.add_argument("--mcpost", action='store_true',
                        help="Make a Monte Carlo sampling of the posterior "
                             "parameter probability distribution function.")
    parser.add_argument("--mc_error", type=str,
                        choices=['const', 'sqrt', 'frac'],
                        help="Error model to use to calculate likelihood ("
                             "required when using --mcpost).")
    parser.add_argument("--mc_steps", type=int, metavar="S",
                        help="Apply the monte carlo step algorithm S times "
                             "(required when using --mcpost.")
    parser.add_argument("--mc_walkers", type=int, metavar="W",
                        help="Use W walkers in the monte carlo search "
                             "(required when using --mcpost.")
    parser.add_argument("--mc_burn", type=int, metavar="B",
                        help="Discard B steps from the chain as burn-in "
                             "(required when using --mcpost).")
    parser.add_argument("--mc_thin", type=int, metavar="T",
                        help="Only keep every Tth step in the chain, "
                             "discarding the rest to reduce autocorrelation ("
                             "required when using --mcpost).")
    parser.add_argument("--mc_threads", type=int, metavar="N",
                        help="Use N threads to run the monte carlo search ("
                             "required when using --mcpost).")
    parser.add_argument("--hideprogress", action='store_false',
                        help="Hides progress bars.")
    parser.add_argument("--rng_seed", type=int,
                        help="Set the RNG seed. If no seed is provided the "
                             "seed will be set automatically from "
                             "nondeterministic OS data.")
    parser.add_argument("--list_models", action='store_true',
                        help="List all available models and their "
                             "parameters.")
    parser.add_argument("--plot_nofit", action='store_true',
                        help="Show data plot without fitting")
    parser.add_argument("--save_output", metavar="PATH",
                        type=str,
                        help="Save output to the given path and don't show "
                             "figures to the screen.")

    args = parser.parse_args(sys_args)

    # Load data file
    print(f'Loading data from {args.tac_path}.')
    tac = tacfit.load_table(args.tac_path)
    print()

    # Report chosen labels:
    if args.time_label in tac:
        print(f'Time label {args.time_label} was found in data file.')
    else:
        print(f'Time label {args.time_label} was not found in data file.')
        exit()

    if args.input_label in tac:
        print(f'Input label {args.input_label} was found in data file.')
    else:
        print(f'Input label {args.input_label} was not found in data file.')
        exit()

    if args.tissue_label in tac:
        print(f'Tissue label {args.tissue_label} was found in data file.')
    else:
        print(f'Tissue label {args.tissue_label} was not found in data file.')
        exit()
    print()

    weights = None
    if args.weight:
        if args.weight in tac:
            print(f'Frame duration label {args.weight} '
                  f'was found in data file.')
            weights = tacfit.calc_weights(tac[args.weight],
                                          tac[args.tissue_label])
            print("Residuals will be weighted with counts")
        else:
            print(f'Frame duration label {args.weight} '
                  f'was not found in data file.')
            exit()
        print()

    # Report chosen tcut/scut and handle int vs list[int] options
    tcut = None
    scut = None
    if args.tcut:
        tcut = args.tcut
        if len(tcut) == 1:
            tcut = tcut[0]
        print(f'Using tcut = {tcut}.')
    elif args.scut:
        scut = args.scut
        print(f'Using scut = {scut}')
    else:
        print("No tcut/scut set, using all data.")
    print()

    # Report chosen tdelay and handle float vs list[float] options
    tdelay = None
    if args.delay:
        tdelay = args.delay
        print(f'Using delay = {tdelay} s.')
    else:
        print("No delay set, using delay = 0 s.")
    print()

    # Possible models to use for fitting:
    models = {
        "nomodel": {
            'irf': lambda x: 1,
            'desc': "Mock model, used when no modelling is required"
        },
        "const": {
            'irf': tacfit.model.irf_const,
            'desc': "Constant (amp)."
        },
        "step2": {
            'irf': tacfit.model.irf_step2,
            'desc':  "Two step functions "
                     "(amp1, extent1, amp2, extent2)."},
        "stepconst": {
            'irf': tacfit.model.irf_stepconst,
            'desc': "Step function followed by constant "
                    "(amp1, extent1, amp2)."},
        "normconst": {
            'irf': tacfit.model.irf_normconst,
            'desc': "Smooth transition to constant "
                    "(amp1, extent1, width1, amp2)."},
        "stepnorm": {
            'irf': tacfit.model.irf_stepnorm,
            'desc': "Step function followed by smooth transition to 0 "
                    "(amp1, extent1, amp2, extent2, width2)."
        },
        "norm2": {
            'irf': tacfit.model.irf_norm2,
            'desc': "Two smooth transitions between amp1, amp2 and 0 "
                    "(amp1, extent1, width1, amp2, extent2, width2)."
        }
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
    if args.param is not None:
        params = _create_params(args.param)
    else:
        params = {}

    if args.exprparam is not None:
        for exprparam in args.exprparam:
            name = exprparam[0]
            expr = exprparam[1]
            params[name] = {'expr': expr}

    print("Parameter settings:")
    if len(params) > 0:
        for param in params:
            print(f'  {param}:')
            if 'value' in params[param]:
                param_value = params[param]['value']
                param_min = params[param]['min']
                param_max = params[param]['max']
                print(f'     value: {param_value}')
                print(f'     min:   {param_min}')
                print(f'     max:   {param_max}')
            else:
                param_expr = params[param]['expr']
                print(f'     expr:  {param_expr}')
    else:
        print("No parameters given.")
    print()

    # Set RNG-seed if chosen
    if args.rng_seed:
        print(f'Setting RNG seed to {args.rng_seed}')
        np.random.seed(args.rng_seed)
    else:
        print("Using nondeterministic RNG seed")
    print()

    # Fit least squares if required
    if args.leastsq:
        print("Starting least squares fitting.")
        output = None
        if args.save_output is not None:
            output = args.save_output

        tacfit.fit_leastsq(tac[args.time_label],
                           tac[args.tissue_label],
                           tac[args.input_label],
                           models[model_str]['irf'],  # type: ignore
                           params,
                           {'tissue': args.tissue_label,
                            'input': args.input_label},
                           weights=weights,
                           output=output,
                           tcut=tcut,
                           scut=scut,
                           confint=args.no_confint,
                           delay=tdelay,
                           progress=args.hideprogress)
        print()

    # Monte Carlo sampling if required
    if args.mcpost:
        print("Starting Monte Calor sampling.")
        print("Monte Carlo parameters:")
        print(f'  errors:  {args.mc_error}')
        print(f'  steps:   {args.mc_steps}')
        print(f'  walkers: {args.mc_walkers}')
        print(f'  burn:    {args.mc_burn}')
        print(f'  thin:    {args.mc_thin}')
        print(f'  threads: {args.mc_threads}')
        output = None
        if args.save_output is not None:
            output = args.save_output
        tacfit.mc_sample(time_data=tac[args.time_label],
                         tissue_data=tac[args.tissue_label],
                         input_data=tac[args.input_label],
                         labels={'tissue': args.tissue_label,
                                 'input': args.input_label},
                         model=models[model_str]['irf'],  # type: ignore
                         params=params,
                         error_model=args.mc_error,
                         nsteps=args.mc_steps,
                         nwalkers=args.mc_walkers,
                         nworkers=args.mc_threads,
                         burn=args.mc_burn,
                         thin=args.mc_thin,
                         output=output,
                         tcut=tcut,
                         scut=scut,
                         delay=tdelay,
                         progress=args.hideprogress)
        print()

    # Plot data if required
    if args.plot_nofit:
        print("Plotting data without fit.")
        output = None
        if args.save_output is not None:
            output = os.path.join(args.save_output, "plot_nofit.png")
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
