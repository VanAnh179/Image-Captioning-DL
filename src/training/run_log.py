"""Copy everything a command prints into a log file under logs/, so a run can be read afterwards."""

import contextlib
import sys
from pathlib import Path


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):  # encoding, isatty and the rest come from the real stream
        return getattr(self.stream, name)


@contextlib.contextmanager
def tee_log(path):
    """Append stdout and stderr to path (created with its folder) while the block runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = out, err
