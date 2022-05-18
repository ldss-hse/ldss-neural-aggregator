import platform
import subprocess
import sys
from pathlib import Path

from constants import ROOT_DIR_PATH


def run_console_tool(tool_path: Path, *args, **kwargs):
    if platform.system() == 'Windows':
        python_executable = ROOT_DIR_PATH / 'venv' / 'Scripts' / 'python.exe'
    else:
        python_executable = ROOT_DIR_PATH / 'venv' / 'bin' / 'python'

    kwargs_processed = []
    for item in kwargs.items():
        kwargs_processed.extend(map(str, item))

    options = [
        str(python_executable), str(tool_path),
        *args,
        *kwargs_processed
    ]

    # print('[SUBPROCESS] {}'.format(' '.join(options)))
    if sys.version_info.minor <= 6:
        return subprocess.run(options)
    else:
        return subprocess.run(options, capture_output=True)
