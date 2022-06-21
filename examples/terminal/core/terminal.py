import importlib
import inspect
import re
import types

from cmd2 import Cmd
from cmd2 import ansi
from typing import Any


from core.state import CFState


class Terminal(Cmd):
    """Terminal class responsible for setting the CLI and loading commands"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            startup_script=".counterfit", allow_cli_args=False, include_ipy=True
        )

        # rename the built-in "set" attribute to "setg"
        setattr(Cmd, "do_setg", Cmd.do_set)
        delattr(Cmd, "do_set")

        self.prompt = "counterfit> "

    def _set_prompt(self):
        """Set the terminal prompt"""
        if not CFState.state().active_target:
            self.prompt = "counterfit> "

        else:
            if not CFState.state().active_target.active_attack:
                self.prompt = f"{CFState.state().active_target.target_name}> "
            else:
                self.prompt = f"{CFState.state().active_target.target_name}>{CFState.state().active_target.active_attack.attack_id}> "

    def default(self, command: str) -> None:
        """Executed when the command given isn't a recognized command implemented by a do_* method.

        Args:
            command (str): command object with parsed input
        """
        print("Command does not exist.\n")
        return

    def precmd(self, line):
        """Run code prior to exe

        Args:
            line ([type]): [description]

        Returns:
            [type]: [description]
        """
        print()
        return line

    def postcmd(self, stop, line):
        """Hook method executed just after a command dispatch is finished."""
        print()

        # Set the prompt to reflect interaction changes
        self._set_prompt()

        # Trigger new choices to populate
        self.load_commands(reload=True)
        return stop

    def load_commands(self, reload=False):
        """Loads all the commands that exists under counterfit.commands sub-package"""
        commands = importlib.import_module("commands")
        if reload:
            commands = importlib.reload(commands)

        for module in inspect.getmembers(commands, inspect.ismodule):
            try:
                command = importlib.import_module(f"commands.{module[0]}")
                if reload:
                    command = importlib.reload(command)

            except ModuleNotFoundError:
                command = None
                continue

            for member in inspect.getmembers(command, inspect.isfunction):
                if "do_" in member[0] or "complete_" in member[0]:
                    setattr(self, member[0], types.MethodType(member[1], self))
                if "finish_" in member[0]:
                    continue

            del command

        del commands

    def pexcept(self, msg: Any, *, end: str = "\n", apply_style: bool = True) -> None:
        """Print an exception
        Args:
            msg (Any): [description]
            end (str, optional): [description]. Defaults to '\n'.
            apply_style (bool, optional): [description]. Defaults to True.
        """
        if isinstance(msg, Exception):
            final_msg = (
                f"EXCEPTION of type '{type(msg).__name__}' occurred with message: {msg}"
            )
        else:
            final_msg = str(msg)

        if apply_style:
            final_msg = ansi.style_error(final_msg)

        if not self.debug and "debug" in self.settables:
            warning = "\n [!] To enable full traceback, run the following command: 'setg debug true'"
            final_msg += ansi.style_warning(warning)

        self.perror(final_msg, end=end, apply_style=False)
