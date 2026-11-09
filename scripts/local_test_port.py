"""Reserve a loopback port and hand its socket to an owned Linux test child."""

import argparse
import errno
import socket


def reserve_loopback_socket(start=18765, *, attempts=100):
    if not 1 <= start <= 65535 or attempts < 1:
        raise ValueError("Invalid starting port or attempt count")
    for port in range(start, min(start + attempts, 65536)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen(128)
            return sock
        except OSError as exc:
            sock.close()
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(errno.EADDRINUSE, "No free loopback port in the bounded test range")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int)
    args = parser.parse_args()
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    import uvicorn

    if args.cpu_threads is not None:
        import torch
        import jeff.server

        original = jeff.server.distributions
        announced = False

        def configured_distributions(model, rows):
            nonlocal announced
            # Configure the inference worker itself, not only the model-loading thread.
            torch.set_num_threads(args.cpu_threads)
            if not announced:
                print(
                    {
                        "inference_cpu_threads": torch.get_num_threads(),
                        "device": model.device_name,
                    },
                    flush=True,
                )
                announced = True
            return original(model, rows)

        jeff.server.distributions = configured_distributions

    with socket.socket(fileno=args.fd) as sock:
        if sock.family != socket.AF_INET or sock.getsockname()[0] != "127.0.0.1":
            raise ValueError("Test server requires a reserved IPv4 loopback socket")
        uvicorn.Server(uvicorn.Config("jeff.server:app", host="127.0.0.1")).run(
            sockets=[sock]
        )


if __name__ == "__main__":
    main()
