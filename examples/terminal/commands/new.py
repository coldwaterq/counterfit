import os
import argparse
from pydoc import locate
from cmd2 import with_argparser
from cmd2 import with_category

from counterfit.reporting import get_data_type_obj_map


from core.state import CFState
from core.config import Config


def get_datatypes():
    return list(get_data_type_obj_map().keys())


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="a name for the new target", required=True)
parser.add_argument(
    "-d",
    "--data_type",
    choices=get_datatypes(),
    help="Send a randomly selected sample to the target model",
)


@with_argparser(parser)
@with_category("Counterfit Commands")
def do_new(self, args: argparse.Namespace) -> None:
    """Optional wizard to aid in creating a new attack target."""

    target_name = args.name.replace(" ", "")

    if not args.data_type:
        target_data_type = ""
    else:
        target_data_type = args.data_type

    if target_name not in os.listdir(Config.targets_path):
        try:
            os.mkdir(f"{Config.targets_path}/{target_name}")
            open(f"{Config.targets_path}/{target_name}/__init__.py", "w").close()
            with open(
                f"{Config.targets_path}/{target_name}/{target_name}.py", "w"
            ) as f:
                f.write(
                    f"""

# Generated by counterfit #

from counterfit.core.targets import CFTarget

class {target_name.capitalize()}(Target):
    target_name = "{target_name}"
    target_data_type = "{target_data_type}"
    target_endpoint = ""
    input_shape = ()
    target_output_classes = []
    target_classifier = ""
    X = []

    def load(self):
        self.X = []

    def predict(self, x):
        return x
"""
                )
        except Exception as e:
            print(f"Failed to write target file: {e}.")
    else:
        print(f"{target_name} already exists. Choose a new name.")

    # Instantiate the new target
    module_path = ".".join(
        f"{Config.targets_path}/{target_name}/{target_name}/{target_name.capitalize()}".split(
            "/"
        )
    )
    new_target = locate(module_path)

    # Add the target to the session
    CFState.state().add_target(target_name, new_target())

    # Load the target
    target = CFState.state().load_target(target_name)

    # Set it as the active target
    CFState.state().set_active_target(target)
