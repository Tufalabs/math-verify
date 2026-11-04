"""Regression coverage for the spawn-safe timeout worker from CAVE."""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.utils import timeout


def in_thread(function, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(function, *args, **kwargs).result(timeout=30)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal routing")
def test_timeout_selects_backend_when_called():
    @timeout(5)
    def process_id():
        return os.getpid()

    assert process_id() == os.getpid()
    assert in_thread(process_id) != os.getpid()
    assert process_id() == os.getpid()


def test_local_callable_results_and_output_survive_worker_transport():
    offset = 3

    @timeout(5)
    def noisy_local_callable():
        os.write(1, b"x" * 131072)
        os.write(2, b"y" * 131072)
        return lambda value: value + offset

    assert in_thread(noisy_local_callable)(2) == 5


@pytest.mark.parametrize("error", [ValueError, TimeoutException])
def test_worker_preserves_exception_type_and_message(error):
    @timeout(5)
    def fail():
        raise error("original failure")

    with pytest.raises(error, match="original failure"):
        in_thread(fail)


@pytest.mark.parametrize("answer,expected", [("0.5", True), ("0.4", False)])
def test_real_parse_and_verify_in_background_thread(answer, expected):
    def grade():
        gold = parse(r"$\frac{1}{2}$", parsing_timeout=5, raise_on_error=True)
        prediction = parse(answer, parsing_timeout=5, raise_on_error=True)
        return verify(gold, prediction, timeout_seconds=5, raise_on_error=True)

    assert in_thread(grade) is expected


@pytest.mark.skipif(os.name != "posix", reason="POSIX process inspection")
def test_timeout_kills_and_reaps_worker(tmp_path):
    pid_path = tmp_path / "pid"

    @timeout(1)
    def sleep():
        pid_path.write_text(str(os.getpid()))
        time.sleep(60)

    with pytest.raises(TimeoutException, match="Operation timed out"):
        in_thread(sleep)
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group inspection")
@pytest.mark.parametrize("crash", [False, True])
def test_timeout_or_crash_terminates_descendants(tmp_path, crash):
    pid_path = tmp_path / "child_pid"

    @timeout(2)
    def spawn_child():
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pid_path.write_text(str(child.pid))
        if crash:
            os._exit(7)
        time.sleep(60)

    error = RuntimeError if crash else TimeoutException
    with pytest.raises(error):
        in_thread(spawn_child)

    # Descendants can briefly remain as zombies until their new parent reaps them.
    status_path = f"/proc/{int(pid_path.read_text())}/status"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            with open(status_path) as stream:
                status = stream.read()
        except FileNotFoundError:
            return
        if "State:\tZ" in status:
            return
        time.sleep(0.01)
    pytest.fail("Timeout worker left a running descendant")
