import argparse
import tacfit
import sys
import importlib.metadata
import time


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
        "step2": {'func': tacfit.model.model_step2,
                  'desc': "Sum of two step functions "
                          "(amp1, extent1, amp2, extent2)"}
    }

    # List models option
    if args.list_models:
        print("List of available models (and parameters):")
        for model in models:
            print(f'-- {model}: {models[model]["desc"]}')
    print()

    # Plot data if required
    if args.plot_nofit:
        print("Plotting data without fit.")
        if args.save_figs is None:
            output = None
        else:
            output = args.save_figs[0]
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
