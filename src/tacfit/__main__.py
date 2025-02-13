import sys
import importlib.metadata
import time
import argparse
import tacfit


def main(sys_args: list[str]):
    # Get version number from pyproject.toml
    __version__ = importlib.metadata.version("tacfit")
    start_time = time.time_ns()

    print("Starting TACFIT", __version__)
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("tac_file", help="Path to TAC-file")
    parser.add_argument("time_label", help="Label of time data")
    parser.add_argument("inp_label", help="Label of input function data")
    parser.add_argument("tis_label", help="Label of tissue data")

    parser.add_argument("--plot", action="store_true",
                        help="Display data without fit")

    args = parser.parse_args(sys_args)

    tac = tacfit.load_table(args.tac_file)

    if args.plot:
        tacfit.plot_data_nofit(tac[args.time_label],
                               tac[args.inp_label],
                               tac[args.tis_label])

    # Report successful end of program
    run_time = (time.time_ns() - start_time) * 1e-9
    print(f'TACFIT finished successfully in {run_time:.1f} seconds.')
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
