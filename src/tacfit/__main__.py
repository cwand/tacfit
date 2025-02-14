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
    parser.add_argument("--plot_nofit", action='store_true',
                        help="Show data plot without fitting")
    parser.add_argument("--leastsq", nargs=1,
                        metavar="MODEL",
                        help="Fit the data to a model")

    args = parser.parse_args(sys_args)

    # Load data file
    tac = tacfit.load_table(args.tac_path)

    # Plot data if required
    if args.plot_nofit:
        tacfit.plot_tac(tac[args.time_label],
                        tac[args.input_label],
                        tac[args.tissue_label],
                        labels={'time': "Time [sec]",
                                'input': args.input_label,
                                'tissue': args.tissue_label,
                                'tacunit': "Act. conc. [Bq/mL]"})

    # Fit data using lmfit least squares if desired
    if args.leastsq is not None:
        models = {
            'step2': tacfit.model.model_step2
        }
        tacfit.fit_leastsq(tac[args.time_label],
                           tac[args.tissue_label],
                           tac[args.input_label],
                           models[args.leastsq[0]],
                           )

    # Report successful end of program
    run_time = (time.time_ns() - start_time) * 1e-9
    print(f'TICTAC finished successfully in {run_time:.1f} seconds.')
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
