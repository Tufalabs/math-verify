# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cloudpickle

from math_verify.errors import TimeoutException

TIMEOUT_WARNING_SHOWN = False
logger = logging.getLogger(__name__)


def _terminate_process(
    process: subprocess.Popen, *, include_process_group: bool = False
) -> None:
    """Stop and reap a timeout worker with bounded waits."""
    windows = sys.platform == "win32"
    if (process.poll() is not None) and (not include_process_group):
        return

    if windows:
        if process.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.5,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
    elif hasattr(os, "killpg"):
        try:
            termination_signal = (
                signal.SIGKILL
                if include_process_group and process.poll() is not None
                else signal.SIGTERM
            )
            os.killpg(
                process.pid,
                termination_signal,
            )
        except OSError:
            pass
    else:
        try:
            process.terminate()
        except OSError:
            pass

    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass

    if not windows and hasattr(os, "killpg") and include_process_group:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
    elif process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass

    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        logger.error("Unable to reap Math-Verify timeout worker %s", process.pid)


def timeout(timeout_seconds: int | None = 10):  # noqa: C901
    """A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
            Defaults to 10 seconds.

    Notes:
        POSIX main-thread calls use a signal alarm. Calls from other threads
        and Windows use a subprocess worker to enforce the timeout.
    """
    if timeout_seconds is None or timeout_seconds <= 0:

        def no_timeout_decorator(func):
            return func

        return no_timeout_decorator

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            main_thread_call = (os.name == "posix") and (
                threading.current_thread() is threading.main_thread()
            )
            if main_thread_call:

                def handler(signum, frame):
                    raise TimeoutException("Operation timed out!")

                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            payload = cloudpickle.dumps((func, args, kwargs))
            with tempfile.TemporaryDirectory(prefix="math-verify-timeout-") as tmp:
                input_path = Path(tmp, "input.bin")
                output_path = Path(tmp, "output.bin")
                input_path.write_bytes(payload)
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "math_verify.timeout_worker",
                        str(input_path),
                        str(output_path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if sys.platform == "win32"
                        else 0
                    ),
                    start_new_session=sys.platform != "win32",
                )

                completed = False
                try:
                    process.wait(timeout=timeout_seconds)
                    completed = True
                except subprocess.TimeoutExpired:
                    raise TimeoutException("Operation timed out!") from None
                finally:
                    if (not completed) or process.returncode:
                        _terminate_process(process, include_process_group=True)

                if process.returncode:
                    raise RuntimeError(
                        f"Timeout worker exited with code {process.returncode}"
                    )
                try:
                    success, value = cloudpickle.loads(output_path.read_bytes())
                except Exception as exc:
                    raise RuntimeError(
                        "Timeout worker returned an invalid response"
                    ) from exc

                if success:
                    return value
                if isinstance(value, BaseException):
                    raise value
                raise RuntimeError("Timeout worker returned an invalid exception")

        return wrapper

    return decorator
