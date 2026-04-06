"""Real-time system resource monitor for SCRIBE pipeline.

Tracks RSS memory, CPU usage, and peak memory while pipeline commands run.
Can be used as a standalone CLI command or as a background thread within
other commands.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TextIO

import psutil


@dataclass
class Snapshot:
    """Single point-in-time resource reading."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    cpu_percent: float
    system_mem_percent: float


@dataclass
class MonitorStats:
    """Accumulated statistics from a monitoring session."""
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    snapshots: list[Snapshot] = field(default_factory=list)

    @property
    def current_rss_mb(self) -> float:
        return self.snapshots[-1].rss_mb if self.snapshots else 0.0

    @property
    def current_cpu_percent(self) -> float:
        return self.snapshots[-1].cpu_percent if self.snapshots else 0.0

    def summary(self) -> str:
        if not self.snapshots:
            return "No data collected."
        last = self.snapshots[-1]
        lines = [
            f"Current RSS:     {last.rss_mb:>8.1f} MB",
            f"Peak RSS:        {self.peak_rss_mb:>8.1f} MB",
            f"Current VMS:     {last.vms_mb:>8.1f} MB",
            f"Peak VMS:        {self.peak_vms_mb:>8.1f} MB",
            f"CPU:             {last.cpu_percent:>7.1f}%",
            f"System memory:   {last.system_mem_percent:>7.1f}%",
            f"Samples:         {len(self.snapshots)}",
        ]
        return "\n".join(lines)


class ResourceMonitor:
    """Background thread that periodically samples process resource usage.

    Usage as context manager::

        with ResourceMonitor(interval=1.0) as mon:
            # ... long-running work ...
            print(mon.stats.peak_rss_mb)

    Or start/stop manually::

        mon = ResourceMonitor()
        mon.start()
        # ... work ...
        mon.stop()
        print(mon.stats.summary())
    """

    def __init__(
        self,
        interval: float = 1.0,
        pid: int | None = None,
        include_children: bool = True,
    ):
        self.interval = interval
        self.pid = pid or os.getpid()
        self.include_children = include_children
        self.stats = MonitorStats()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> Snapshot | None:
        try:
            proc = psutil.Process(self.pid)
            mem = proc.memory_info()
            rss = mem.rss
            vms = mem.vms

            # Include child processes (scanpy spawns subprocesses sometimes)
            if self.include_children:
                for child in proc.children(recursive=True):
                    try:
                        cmem = child.memory_info()
                        rss += cmem.rss
                        vms += cmem.vms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            rss_mb = rss / (1024 * 1024)
            vms_mb = vms / (1024 * 1024)
            cpu = proc.cpu_percent(interval=None)
            sys_mem = psutil.virtual_memory().percent

            snap = Snapshot(
                timestamp=time.time(),
                rss_mb=rss_mb,
                vms_mb=vms_mb,
                cpu_percent=cpu,
                system_mem_percent=sys_mem,
            )
            return snap
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def _run(self) -> None:
        # Prime cpu_percent (first call always returns 0)
        try:
            psutil.Process(self.pid).cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        while not self._stop_event.is_set():
            snap = self._sample()
            if snap is not None:
                self.stats.snapshots.append(snap)
                self.stats.peak_rss_mb = max(self.stats.peak_rss_mb, snap.rss_mb)
                self.stats.peak_vms_mb = max(self.stats.peak_vms_mb, snap.vms_mb)
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def live_monitor(
    pid: int | None = None,
    interval: float = 2.0,
    include_children: bool = True,
    output: TextIO | None = None,
) -> None:
    """Print live resource usage to the terminal until interrupted.

    This is the function backing the ``python run.py monitor`` CLI command.
    It prints a refreshing status line showing RSS, peak RSS, CPU%, and
    system memory usage.

    Args:
        pid: Process ID to monitor. Defaults to current process.
        interval: Seconds between samples.
        include_children: Include child process memory.
        output: File-like object for output. Defaults to sys.stderr.
    """
    out = output or sys.stderr
    mon = ResourceMonitor(interval=interval, pid=pid or os.getpid(),
                          include_children=include_children)

    # Handle Ctrl+C gracefully
    interrupted = threading.Event()

    def _sigint(signum, frame):
        interrupted.set()

    prev_handler = signal.signal(signal.SIGINT, _sigint)

    mon.start()
    out.write("SCRIBE Resource Monitor — press Ctrl+C to stop\n")
    out.write(f"Monitoring PID {mon.pid} (interval: {interval}s)\n")
    out.write("-" * 60 + "\n")

    try:
        while not interrupted.is_set():
            time.sleep(interval)
            snap = mon.stats.snapshots[-1] if mon.stats.snapshots else None
            if snap is None:
                continue

            line = (
                f"RSS: {snap.rss_mb:>7.1f} MB | "
                f"Peak: {mon.stats.peak_rss_mb:>7.1f} MB | "
                f"CPU: {snap.cpu_percent:>5.1f}% | "
                f"Sys Mem: {snap.system_mem_percent:>5.1f}%"
            )
            out.write(f"\r{line}")
            out.flush()
    finally:
        mon.stop()
        signal.signal(signal.SIGINT, prev_handler)
        out.write("\n" + "-" * 60 + "\n")
        out.write(mon.stats.summary() + "\n")


def monitor_command(
    pid: int | None = None,
    interval: float = 2.0,
    log_path: str | None = None,
) -> None:
    """Entry point for the CLI ``monitor`` command.

    If *log_path* is given, appends CSV rows to a file alongside live output.

    Args:
        pid: Process ID to monitor. Defaults to current process.
        interval: Seconds between samples.
        log_path: Optional CSV file to append resource logs to.
    """
    target_pid = pid or os.getpid()
    mon = ResourceMonitor(interval=interval, pid=target_pid, include_children=True)

    log_file = None
    if log_path:
        log_file = open(log_path, "a")
        # Write header if file is empty
        if os.path.getsize(log_path) == 0 or log_file.tell() == 0:
            log_file.write("timestamp,rss_mb,vms_mb,cpu_percent,system_mem_percent\n")

    interrupted = threading.Event()

    def _sigint(signum, frame):
        interrupted.set()

    prev_handler = signal.signal(signal.SIGINT, _sigint)
    mon.start()

    print(f"SCRIBE Resource Monitor — PID {target_pid}, interval {interval}s")
    print(f"System: {psutil.virtual_memory().total / (1024**3):.1f} GB total RAM")
    if log_path:
        print(f"Logging to: {log_path}")
    print("-" * 60)

    try:
        while not interrupted.is_set():
            time.sleep(interval)
            if not mon.stats.snapshots:
                continue
            snap = mon.stats.snapshots[-1]
            line = (
                f"RSS: {snap.rss_mb:>7.1f} MB | "
                f"Peak: {mon.stats.peak_rss_mb:>7.1f} MB | "
                f"CPU: {snap.cpu_percent:>5.1f}% | "
                f"Sys: {snap.system_mem_percent:>5.1f}%"
            )
            sys.stdout.write(f"\r{line}")
            sys.stdout.flush()

            if log_file:
                log_file.write(
                    f"{snap.timestamp:.2f},{snap.rss_mb:.1f},{snap.vms_mb:.1f},"
                    f"{snap.cpu_percent:.1f},{snap.system_mem_percent:.1f}\n"
                )
                log_file.flush()
    finally:
        mon.stop()
        signal.signal(signal.SIGINT, prev_handler)
        if log_file:
            log_file.close()
        print("\n" + "-" * 60)
        print(mon.stats.summary())
