"""
Minimal fallback implementation of the ``line_profiler`` API used in the project.

This is a pure-Python line profiler that provides a subset of the real package's
interface (``LineProfiler`` with ``add_function``, callable wrappers, and
``print_stats``). It is sufficient for collecting line-level timings when the
compiled ``line_profiler`` wheels are unavailable in the current environment.
"""

from __future__ import annotations

import linecache
import sys
import time
from collections import defaultdict, OrderedDict
from types import CodeType, FrameType
from typing import Callable, Dict, Iterable, Optional, Tuple


class LineProfiler:
    """Simple line profiler compatible with the subset of the third-party API."""

    def __init__(self) -> None:
        self._code_map: Dict[CodeType, Dict[int, float]] = {}
        self._enabled = 0
        self._frame_last: Dict[FrameType, Tuple[float, int]] = {}

    # ------------------------------------------------------------------ #
    # Registration & tracing control
    # ------------------------------------------------------------------ #
    def add_function(self, func: Callable) -> None:
        code = getattr(func, "__code__", None)
        if code is None:
            raise TypeError(f"Expected function with __code__, got {func!r}")
        if code not in self._code_map:
            self._code_map[code] = defaultdict(float)

    def add_functions(self, funcs: Iterable[Callable]) -> None:
        for func in funcs:
            self.add_function(func)

    def __call__(self, func: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            self.enable_by_count()
            try:
                return func(*args, **kwargs)
            finally:
                self.disable_by_count()

        return wrapped

    def enable_by_count(self) -> None:
        if self._enabled == 0:
            sys.settrace(self._trace_dispatch)
        self._enabled += 1

    def disable_by_count(self) -> None:
        if self._enabled == 0:
            return
        self._enabled -= 1
        if self._enabled == 0:
            sys.settrace(None)
            self._frame_last.clear()

    # ------------------------------------------------------------------ #
    # Tracing internals
    # ------------------------------------------------------------------ #
    def _trace_dispatch(self, frame: FrameType, event: str, arg):
        if event == "call":
            code = frame.f_code
            if code in self._code_map:
                return self._trace_lines
        return None

    def _trace_lines(self, frame: FrameType, event: str, arg):
        code = frame.f_code
        timings = self._code_map.get(code)
        if timings is None:
            return None

        if event == "line":
            now = time.perf_counter()
            last = self._frame_last.get(frame)
            if last is not None:
                last_time, last_line = last
                timings[last_line] += now - last_time
            self._frame_last[frame] = (now, frame.f_lineno)
        elif event == "return":
            now = time.perf_counter()
            last = self._frame_last.pop(frame, None)
            if last is not None:
                last_time, last_line = last
                timings[last_line] += now - last_time
        return self._trace_lines

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def print_stats(
        self,
        stream: Optional[Callable[[str], None]] = None,
        stripzeros: bool = True,
        output: Optional[Callable[[str], None]] = None,
    ) -> None:
        if output is not None:
            emitter: Callable[[str], None] = output
        elif stream is not None:
            emitter = stream.write if hasattr(stream, "write") else stream
        else:
            emitter = print

        for code, timings in self._code_map.items():
            filename = code.co_filename
            func_name = code.co_name
            header = (
                f"File: {filename}\nFunction: {func_name} at line {code.co_firstlineno}"
            )
            emitter(header)
            emitter("-" * len(header))

            if not timings:
                emitter("  No recorded line timings.\n")
                continue

            ordered = OrderedDict(sorted(timings.items()))
            total_time = sum(ordered.values())

            line_number_width = max(len(str(line)) for line in ordered)
            emitter(
                f"{'Line':>{line_number_width}}    Time (s)    % Time    Line Contents"
            )
            emitter("-" * (line_number_width + 44))

            for line_no, elapsed in ordered.items():
                if stripzeros and elapsed == 0.0:
                    continue
                percent = (elapsed / total_time) * 100 if total_time else 0.0
                text = linecache.getline(filename, line_no).rstrip()
                emitter(
                    f"{line_no:>{line_number_width}}    {elapsed:>8.6f}    {percent:6.2f}    {text}"
                )
            emitter(f"Total time: {total_time:.6f} s\n")

    def get_stats(self):
        """
        Return profiling statistics as an iterable of (code, timings) tuples.

        Each ``timings`` entry is a dictionary mapping line numbers to elapsed
        wall-time in seconds.
        """
        return [(code, dict(timings)) for code, timings in self._code_map.items()]


__all__ = ["LineProfiler"]
