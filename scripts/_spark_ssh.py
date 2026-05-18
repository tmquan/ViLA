"""Tiny SSH helper for the DGX Spark host (Tailscale-routed).

Usage::

    python scripts/_spark_ssh.py "uname -a; nvidia-smi --version"

Reads the password from ``SPARK_SSH_PASS`` if set; otherwise prompts.
Connection details are hard-coded for this project; rotate them when
the credentials change.

The script streams stdout/stderr live so long-running commands (e.g. a
NIM container pull) print progress as they go.
"""

from __future__ import annotations

import os
import sys

import paramiko

HOST = "spark-5e98.tailf9a251.ts.net"
USER = "quantm"
PORT = 22


def run_remote(cmd: str, password: str, *, timeout: float = 60.0) -> int:
    """Run ``cmd`` on the Spark; stream output; return exit code."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=HOST,
        port=PORT,
        username=USER,
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=timeout,
    )
    try:
        # Request a TTY so progress bars (e.g. docker pull) refresh.
        transport = client.get_transport()
        assert transport is not None
        chan = transport.open_session()
        chan.get_pty()
        chan.exec_command(cmd)
        # Stream stdout + stderr interleaved.
        while True:
            if chan.recv_ready():
                sys.stdout.write(chan.recv(4096).decode(errors="replace"))
                sys.stdout.flush()
            if chan.recv_stderr_ready():
                sys.stderr.write(chan.recv_stderr(4096).decode(errors="replace"))
                sys.stderr.flush()
            if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                break
        return chan.recv_exit_status()
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: _spark_ssh.py <command...>", file=sys.stderr)
        return 2
    cmd = " ".join(argv)
    password = os.environ.get("SPARK_SSH_PASS", "")
    if not password:
        print("error: set SPARK_SSH_PASS env var with the Spark login password",
              file=sys.stderr)
        return 2
    return run_remote(cmd, password)


if __name__ == "__main__":
    sys.exit(main())
