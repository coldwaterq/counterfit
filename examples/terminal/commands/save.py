import argparse
from cmd2 import with_argparser
from cmd2 import with_category

from core.state import CFState

from counterfit.core.errors import CFErrors

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--parameters", action="store_true", help="Save the parameters for an attack"
)
parser.add_argument(
    "-r",
    "--results",
    action="store_true",
    help="Save the results and metadata for an attack",
)


@with_argparser(parser)
@with_category("Counterfit Commands")
def do_save(self, args: argparse.Namespace) -> None:
    """Save results or parameters to disk.

    Args:
        results (bool): save the results from `cfattack.results`.
        parameters (bool): Save the parameters used for the attack.
    """

    if not CFState.state().active_target:
        print("Not interacting with a target. Set the active target with `interact`.")
        return

    if not CFState.state().active_target.active_attack:
        print("No active target. Set the active target with `use`.")
        return

    results_folder = CFState.state().active_target.active_attack.get_results_folder()

    if args.parameters:
        CFState.state().active_target.active_attack.options.save_options(
            f"{results_folder}/params.json"
        )
        print(f"Successfully wrote {results_folder}/params.json")

    if args.results:
        CFState.state().active_target.active_attack.save_run_summary(
            f"{results_folder}/run_summary.json"
        )
        print(f"Successfully wrote {results_folder}/run_summary.json")
