import errno
import os
import socket
import subprocess
import sys

import pytest

from scripts.local_test_port import reserve_loopback_socket


def test_occupied_starting_port_is_skipped_and_reservation_stays_open():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        with reserve_loopback_socket(port) as chosen:
            assert chosen.getsockname()[1] > port
            with socket.socket() as competitor, pytest.raises(OSError) as caught:
                competitor.bind(chosen.getsockname())
            assert caught.value.errno == errno.EADDRINUSE


def test_busy_range_is_bounded_and_other_errors_are_not_hidden(monkeypatch):
    calls = []

    class Busy:
        def bind(self, address):
            calls.append(address)
            raise OSError(errno.EADDRINUSE, "occupied")

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args: Busy())
    with pytest.raises(OSError):
        reserve_loopback_socket(18765, attempts=3)
    assert calls == [("127.0.0.1", port) for port in (18765, 18766, 18767)]

    def denied(*args):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(socket, "socket", denied)
    with pytest.raises(PermissionError):
        reserve_loopback_socket()


@pytest.mark.skipif(
    os.name != "posix", reason="POSIX fd handoff for local Linux test services"
)
def test_child_inherits_reserved_socket_without_rebinding():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    reserved = reserve_loopback_socket(port)
    address = reserved.getsockname()
    fd = reserved.fileno()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,sys; s=socket.socket(fileno=int(sys.argv[1])); c,a=s.accept(); c.sendall(b'owned'); c.close(); s.close()",
            str(fd),
        ],
        pass_fds=(fd,),
    )
    reserved.close()
    try:
        with socket.create_connection(address, timeout=3) as client:
            assert client.recv(5) == b"owned"
        assert child.wait(timeout=3) == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=3)
