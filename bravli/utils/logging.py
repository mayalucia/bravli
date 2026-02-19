"""A print-based logger for interactive scientific computing.

Standard Python logging disappears in Jupyter notebooks unless carefully
configured. This module provides a simple alternative: print to stdout
with timestamps and level labels. Unsophisticated, but visible.

Usage:
    from bravli.utils import get_logger
    log = get_logger("my_module")
    log.info("Loading %s neurons", 1000)
"""

import sys
from datetime import datetime


def get_logger(name, out=None):
    """Create a print-based logger.

    Parameters
    ----------
    name : str
        Logger name, displayed in every message header.
    out : file-like, optional
        Additional output stream (e.g., an open log file).

    Returns
    -------
    callable
        A log function with .debug, .info, .warning, .error methods.
    """
    prefix = f"bravli:{name}"
    line_length = 72
    outputs = [sys.stdout] + ([out] if out else [])

    def _header(level):
        now = datetime.now().strftime("%H:%M:%S")
        for dest in outputs:
            print(f"{'_' * line_length}", file=dest)
            print(f"{prefix} {level} [{now}]", file=dest)

    def log(level, msg, args):
        _header(level)
        for dest in outputs:
            try:
                print(msg % args, file=dest)
            except TypeError:
                print(msg, file=dest)

    log.debug = lambda msg, *args: log("DEBUG", msg, args)
    log.info = lambda msg, *args: log("INFO", msg, args)
    log.warning = lambda msg, *args: log("WARNING", msg, args)
    log.error = lambda msg, *args: log("ERROR", msg, args)

    return log
