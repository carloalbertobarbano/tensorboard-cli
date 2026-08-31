#!/usr/bin/env python3
import argparse
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

EVENT_FILE_PREFIX = "events.out.tfevents."

try:
    import tty as _tty
    import termios as _termios
    import select as _select
    _HAS_TTY = True
except ImportError:  # pragma: no cover - Windows or restricted env
    _HAS_TTY = False

_PALETTE_RGB: List[Tuple[int, int, int]] = [
    (220, 80,  80),
    (80,  200, 80),
    (80,  120, 220),
    (220, 160, 50),
    (180, 80,  220),
    (60,  200, 200),
    (220, 220, 80),
    (220, 120, 180),
]

_ANSI_RESET = "\033[0m"
_ANSI_BOLD  = "\033[1m"
_ANSI_DIM   = "\033[2m"


def _ansi_fg(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _dim_color(r: int, g: int, b: int, dark_mode: bool = False, factor: float = 0.35) -> Tuple[int, int, int]:
    bg = 10 if dark_mode else 230
    return (
        int(bg + (r - bg) * factor),
        int(bg + (g - bg) * factor),
        int(bg + (b - bg) * factor),
    )


@dataclass
class ScalarPoint:
    step: int
    value: float
    wall_time: float


@dataclass
class InteractiveState:
    legend_position: str = "top"
    cursor_idx: int = 0
    highlighted_runs: Set[str] = field(default_factory=set)
    dark_mode: bool = False

    def copy(self) -> "InteractiveState":
        return InteractiveState(
            legend_position=self.legend_position,
            cursor_idx=self.cursor_idx,
            highlighted_runs=set(self.highlighted_runs),
            dark_mode=self.dark_mode,
        )


def _discover_new_runs(
    logdir: Path,
    runs_filter: Optional[str],
    current_runs: Sequence[Path],
) -> List[Path]:
    """Re-scan ``logdir`` and return runs not already in ``current_runs``.

    Used by the web server's background reload to pick up runs created after
    the server started. ``runs_filter`` applies the same ``--runs`` wildcard
    patterns (if any) so an unrelated new run doesn't sneak in. Existing runs
    are never removed — only new ones are returned — so a run the user is
    viewing is never dropped out from under them.
    """
    discovered = discover_runs(logdir)
    if runs_filter:
        discovered = filter_runs_by_patterns(discovered, runs_filter)
    existing = {str(p) for p in current_runs}
    return [r for r in discovered if str(r) not in existing]


def discover_runs(logdir: Path) -> List[Path]:
    runs = []
    seen = set()
    for root, _, files in os.walk(logdir):
        if any(name.startswith(EVENT_FILE_PREFIX) for name in files):
            run_path = Path(root)
            if run_path not in seen:
                seen.add(run_path)
                runs.append(run_path)
    return sorted(runs)


def filter_runs_by_patterns(
    runs: Sequence[Path], patterns_raw: str
) -> List[Path]:
    """Keep only runs whose name or path matches one of the comma-separated
    wildcard patterns (fnmatch-style, e.g. ``*exp1*,2024*``).

    A bare ``*`` or ``all`` (case-insensitive) matches every run, so callers can
    pass the same value they'd give the index-based selector.
    """
    import fnmatch

    tokens = [t.strip() for t in patterns_raw.split(",") if t.strip()]
    if any(tok.lower() in {"all", "*"} for tok in tokens):
        return list(runs)
    kept: List[Path] = []
    for run in runs:
        name = run.name
        path_str = str(run)
        if any(fnmatch.fnmatch(name, tok) or fnmatch.fnmatch(path_str, tok) for tok in tokens):
            kept.append(run)
    return kept


def _event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "tensorboard is required. Install it with: pip install 'tensorboard>=2.0'"
        ) from exc
    return EventAccumulator


def load_scalars(
    runs: Sequence[Path], loader_cls=None
) -> Dict[str, Dict[str, List[ScalarPoint]]]:
    loader = loader_cls or _event_accumulator()
    loaded: Dict[str, Dict[str, List[ScalarPoint]]] = {}
    for run in runs:
        acc = loader(str(run))
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        loaded[str(run)] = {}
        for tag in tags:
            points = [
                ScalarPoint(
                    step=int(event.step),
                    value=float(event.value),
                    wall_time=float(event.wall_time),
                )
                for event in acc.Scalars(tag)
            ]
            loaded[str(run)][tag] = points
    return loaded


_TF_RECORD_HEADER_BYTES = 12  # uint64 data_length + uint32 masked CRC of length
_TF_RECORD_FOOTER_BYTES = 4   # uint32 masked CRC of data


def _count_tfrecord_events(path: str) -> int:
    """Count TF records using header seeks only — no protobuf parsing."""
    count = 0
    with open(path, "rb") as fh:
        while True:
            header = fh.read(_TF_RECORD_HEADER_BYTES)
            if len(header) < _TF_RECORD_HEADER_BYTES:
                break
            (data_len,) = struct.unpack("<Q", header[:8])
            fh.seek(data_len + _TF_RECORD_FOOTER_BYTES, 1)
            count += 1
    return count


def _scan_step_info(path: str) -> Tuple[int, int, int]:
    """Parse the first two logged steps to estimate file characteristics.

    Returns (records_per_step, first_step, step_interval) where records_per_step
    is the number of scalar records per step, first_step is the first logged step
    value, and step_interval is the gap between consecutive logged steps.
    """
    from tensorboard.compat.proto.event_pb2 import Event

    tags_in_first_step: Set[str] = set()
    first_step: Optional[int] = None
    with open(path, "rb") as fh:
        for _ in range(10_000):
            header = fh.read(_TF_RECORD_HEADER_BYTES)
            if len(header) < _TF_RECORD_HEADER_BYTES:
                break
            (data_len,) = struct.unpack("<Q", header[:8])
            data = fh.read(data_len)
            fh.read(_TF_RECORD_FOOTER_BYTES)
            ev = Event()
            ev.ParseFromString(data)
            for sv in ev.summary.value:
                if sv.HasField("simple_value"):
                    step = ev.step
                    if first_step is None:
                        first_step = step
                        tags_in_first_step.add(sv.tag)
                    elif step == first_step:
                        tags_in_first_step.add(sv.tag)
                    else:
                        rps = max(len(tags_in_first_step), 1)
                        interval = max(step - first_step, 1)
                        return rps, first_step, interval
    rps = max(len(tags_in_first_step), 1)
    return rps, first_step or 0, 1


def _parse_event_step(data: bytes) -> Optional[int]:
    """Extract step from raw Event proto bytes without full protobuf parsing.

    TensorFlow always writes Event fields in field-number order:
      field 1 wall_time (fixed64): tag byte 0x09 + 8 data bytes
      field 2 step (varint):       tag byte 0x10 + varint bytes
    Returns None on unexpected format, 0 for events without a step field.
    """
    if len(data) < 9 or data[0] != 0x09:
        return None
    pos = 9  # skip wall_time (1 tag byte + 8 data bytes)
    if pos >= len(data) or data[pos] != 0x10:
        return 0  # no step field (e.g. FileVersion events)
    pos += 1
    step, shift = 0, 0
    end = min(pos + 10, len(data))  # varint is at most 10 bytes
    while pos < end:
        b = data[pos]; pos += 1
        step |= (b & 0x7F) << shift; shift += 7
        if not (b & 0x80):
            return step
    return None


def _iter_tfrecord_stride_by_step(
    path: str,
    stride: int,
    tail_records: int,
    total_records: int,
    first_step: int = 0,
    step_interval: int = 1,
    records_per_step: int = 0,
) -> Iterator[bytes]:
    """Yield raw event bytes from a TF record file with step-index subsampling.

    Reads all records sequentially (cache-friendly, no seek-syscall overhead)
    and groups them by step (records for a given step are contiguous in the
    file). A step's records are kept when any of these holds:

      * the record falls in the last ``tail_records`` positions (full fidelity);
      * the step carries a non-standard number of scalar records
        (``records_per_step`` is the count at a typical/dense step) — this
        preserves metrics logged on a sparser cadence than the dense ones,
        which a plain stride would otherwise drop entirely; or
      * the step index ``(step - first_step) // step_interval`` is divisible by
        ``stride`` (the 1-in-S decimation for the dense metrics).

    This makes stride=S mean "keep 1 in S dense logged steps" while still
    keeping every point of any metric logged on a different cadence.
    """
    tail_start = max(0, total_records - tail_records)
    stride = max(1, stride)
    interval = step_interval or 1

    def _keep_step(step: Optional[int], count: int) -> bool:
        # A step whose record count differs from the dense norm carries
        # sparsely-logged (or missing) metrics — keep it in full so those
        # metrics aren't lost to the stride decimation.
        if records_per_step and count != records_per_step:
            return True
        if step is None:
            return True
        idx = (step - first_step) // interval
        return idx % stride == 0

    with open(path, "rb") as fh:
        # Buffer records of the current step until the step changes, so the
        # keep decision can use the step's full record count. One dense step is
        # at most a few hundred small records, so this is cheap.
        buf: List[Tuple[int, bytes]] = []
        cur_step: Optional[int] = None
        idx = 0
        for _ in range(total_records):
            header = fh.read(_TF_RECORD_HEADER_BYTES)
            if len(header) < _TF_RECORD_HEADER_BYTES:
                break
            (data_len,) = struct.unpack("<Q", header[:8])
            data = fh.read(data_len)
            fh.read(_TF_RECORD_FOOTER_BYTES)

            step = _parse_event_step(data)
            if step != cur_step and buf:
                keep = _keep_step(cur_step, len(buf))
                for gidx, d in buf:
                    if keep or gidx >= tail_start:
                        yield d
                buf = []
            cur_step = step
            buf.append((idx, data))
            idx += 1

        if buf:
            keep = _keep_step(cur_step, len(buf))
            for gidx, d in buf:
                if keep or gidx >= tail_start:
                    yield d


def _find_event_files(run_dir: Path) -> List[str]:
    return sorted(
        str(p) for p in run_dir.iterdir()
        if p.name.startswith(EVENT_FILE_PREFIX)
    )


def _run_created_time(run_dir: Path) -> float:
    """Return the run's creation time (unix seconds).

    TensorBoard event filenames embed the creation timestamp right after the
    prefix (events.out.tfevents.<unixtime>.<host>...), giving a reliable,
    cross-platform value. Use the earliest across event files; fall back to the
    filesystem mtime if no timestamp can be parsed.
    """
    event_files = _find_event_files(run_dir)
    times: List[float] = []
    for path in event_files:
        name = Path(path).name
        stamp = name[len(EVENT_FILE_PREFIX):].split(".", 1)[0]
        try:
            times.append(float(stamp))
        except ValueError:
            continue
    if times:
        return min(times)
    try:
        if event_files:
            return min(os.path.getmtime(p) for p in event_files)
        return os.path.getmtime(run_dir)
    except OSError:
        return 0.0


def limit_to_most_recent_runs(
    runs: Sequence[Path], last_n: Optional[int]
) -> List[Path]:
    """Keep only the ``last_n`` most recently created runs.

    Recency is measured by :func:`_run_created_time` (the timestamp embedded in
    each event filename). Runs are returned in their original discovery order
    so downstream index-based and pattern-based selection stays consistent.
    ``last_n`` of ``None`` or ``<= 0`` disables the limit.
    """
    if not last_n or last_n <= 0 or last_n >= len(runs):
        return list(runs)
    ranked = sorted(runs, key=_run_created_time, reverse=True)
    kept = set(ranked[:last_n])
    return [r for r in runs if r in kept]


def load_scalars_fast(
    runs: Sequence[Path],
    stride: int = 10,
    tail: int = 500,
) -> Dict[str, Dict[str, List[ScalarPoint]]]:
    """
    Like load_scalars but uses stride subsampling for much faster loading.

    For each event file: one seek-only pass counts records, then one sequential
    read pass extracts the step field cheaply and only does a full protobuf
    parse for steps matching step % stride == 0 plus the last
    (tail * tags_per_step) records at full fidelity.
    """
    from tensorboard.compat.proto.event_pb2 import Event

    loaded: Dict[str, Dict[str, List[ScalarPoint]]] = {}
    for run in runs:
        run_data: Dict[str, List[ScalarPoint]] = {}
        for event_file in _find_event_files(run):
            total = _count_tfrecord_events(event_file)
            if total == 0:
                continue
            rps, first_step, step_interval = _scan_step_info(event_file)
            tail_records = tail * rps

            for raw in _iter_tfrecord_stride_by_step(
                event_file, stride, tail_records, total, first_step, step_interval, rps
            ):
                ev = Event()
                ev.ParseFromString(raw)
                for sv in ev.summary.value:
                    if sv.HasField("simple_value"):
                        run_data.setdefault(sv.tag, []).append(ScalarPoint(
                            step=int(ev.step),
                            value=float(sv.simple_value),
                            wall_time=float(ev.wall_time),
                        ))

        loaded[str(run)] = run_data
    return loaded


def parse_selection(raw: str, total: int) -> List[int]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if any(token.lower() in {"all", "*"} for token in tokens):
        return list(range(total))
    selected = []
    for token in tokens:
        idx = int(token) - 1
        if idx < 0 or idx >= total:
            raise ValueError(f"Invalid index: {token}")
        selected.append(idx)
    if not selected:
        raise ValueError("No valid selections provided")
    return sorted(set(selected))


def _is_index_token(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def resolve_run_selection(runs: Sequence[Path], run_arg: str) -> List[Path]:
    """Resolve a comma-separated ``--runs`` argument for CLI mode.

    Each token is one of:

    * the keyword ``all`` or ``*`` (case-insensitive) — every run;
    * a 1-based run index (e.g. ``1``, ``3``) — the run at that position in
      the sorted discovery list;
    * an fnmatch wildcard pattern (e.g. ``*exp1*``, ``2024*``) matched
      against the run name **and** full path, exactly like ``--web`` mode.

    Tokens of different kinds may be mixed, e.g. ``1,3,*exp*``. A pattern
    that matches no run raises ``ValueError`` so a typo is never silently
    dropped. The result is returned in discovery order with duplicates
    removed.
    """
    import fnmatch

    tokens = [t.strip() for t in run_arg.split(",") if t.strip()]
    if any(tok.lower() in {"all", "*"} for tok in tokens):
        return list(runs)
    chosen = set()
    for tok in tokens:
        if _is_index_token(tok):
            idx = int(tok) - 1
            if idx < 0 or idx >= len(runs):
                raise ValueError(f"Invalid run index: {tok}")
            chosen.add(runs[idx])
        else:
            matches = [r for r in runs
                       if fnmatch.fnmatch(r.name, tok) or fnmatch.fnmatch(str(r), tok)]
            if not matches:
                raise ValueError(f"No runs match pattern: {tok!r}")
            chosen.update(matches)
    if not chosen:
        raise ValueError("No valid selections provided")
    return [r for r in runs if r in chosen]


def prompt_run_selection(runs: Sequence[Path]) -> List[Path]:
    print("Available runs:")
    for idx, run in enumerate(runs, start=1):
        print(f"  [{idx}] {run}")
    while True:
        raw = input("Select runs (e.g. 1,2, *exp* or all/*): ")
        try:
            return resolve_run_selection(runs, raw)
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


def prompt_metric_selection(metrics: Sequence[str]) -> str:
    print("Available scalar metrics:")
    for idx, metric in enumerate(metrics, start=1):
        print(f"  [{idx}] {metric}")
    while True:
        raw = input("Select metric index: ").strip()
        try:
            idx = int(raw) - 1
            if idx < 0 or idx >= len(metrics):
                raise ValueError
            return metrics[idx]
        except ValueError:
            print("Invalid metric selection")


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def render_sparklines(series: Dict[str, List[ScalarPoint]]) -> str:
    if not any(series.values()):
        return "No points to plot."
    max_name_len = max(len(name) for name in series)
    lines = []
    for run_name, points in series.items():
        if not points:
            lines.append(f"  {run_name:<{max_name_len}}  (no data)")
            continue
        values = [p.value for p in points]
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        spark = "".join(
            _SPARK_CHARS[min(7, int((v - lo) / span * 7.9999))] for v in values
        )
        last = points[-1]
        lines.append(
            f"  {run_name:<{max_name_len}}  {spark}"
            f"  step={last.step}  last={last.value:.4g}"
            f"  min={lo:.4g}  max={hi:.4g}"
        )
    return "\n".join(lines)


def _render_legend(
    run_names: List[str],
    last_values: Dict[str, float],
    state: InteractiveState,
    last_steps: Optional[Dict[str, int]] = None,
) -> str:
    has_highlight = bool(state.highlighted_runs)
    lines = []
    for i, name in enumerate(run_names):
        r, g, b = _PALETTE_RGB[i % len(_PALETTE_RGB)]
        is_cursor = i == state.cursor_idx
        is_highlighted = name in state.highlighted_runs
        cursor_str = "> " if is_cursor else "  "

        if is_highlighted:
            fmt = _ANSI_BOLD + _ansi_fg(r, g, b)
        elif has_highlight:
            dr, dg, db = _dim_color(r, g, b, dark_mode=state.dark_mode)
            fmt = (_ANSI_DIM if state.dark_mode else "") + _ansi_fg(dr, dg, db)
        else:
            fmt = _ansi_fg(r, g, b)

        last_val = last_values.get(name, float("nan"))
        step_str = f"  step={last_steps[name]}" if last_steps and name in last_steps else ""
        lines.append(f"  {cursor_str}{fmt}{name}{step_str}  {last_val:.6g}{_ANSI_RESET}")
    return "\n".join(lines)


def render_plot_plotext(
    series: Dict[str, List[ScalarPoint]],
    metric: str = "value",
    width: int = 80,
    height: int = 24,
    state: Optional[InteractiveState] = None,
) -> str:
    try:
        import plotext as plt  # type: ignore
    except ImportError:
        return "plotext not installed. Run: pip install 'plotext>=5.0'"
    if not any(series.values()):
        return "No points to plot."

    plt.clf()
    plt.plot_size(width, height)
    plt.title(metric)
    plt.xlabel("step")
    plt.ylabel("value")
    if state is not None and state.dark_mode:
        plt.canvas_color("black")
        plt.axes_color("black")
        plt.ticks_color("white")

    has_highlight = state is not None and bool(state.highlighted_runs)
    run_names = list(series.keys())
    last_values: Dict[str, float] = {}
    last_steps: Dict[str, int] = {}

    for i, (run_name, points) in enumerate(series.items()):
        if not points:
            continue
        last_val = points[-1].value
        last_values[run_name] = last_val
        last_steps[run_name] = points[-1].step
        r, g, b = _PALETTE_RGB[i % len(_PALETTE_RGB)]

        if has_highlight and run_name not in state.highlighted_runs:  # type: ignore[union-attr]
            color: Tuple[int, int, int] = _dim_color(r, g, b, dark_mode=state.dark_mode)  # type: ignore[union-attr]
        else:
            color = (r, g, b)

        if state is not None:
            plt.plot([p.step for p in points], [p.value for p in points], color=color)
        else:
            plt.plot(
                [p.step for p in points],
                [p.value for p in points],
                label=f"{run_name}  {last_val:.6g}",
                color=color,
            )

    plot_str = plt.build()

    if state is None:
        return plot_str

    legend_str = _render_legend(run_names, last_values, state, last_steps=last_steps)
    if state.legend_position == "bottom":
        return plot_str + "\n" + legend_str
    return legend_str + "\n" + plot_str


def render_plot_ascii(series: Dict[str, List[ScalarPoint]], height: int = 12) -> str:
    try:
        import asciichartpy  # type: ignore
    except ImportError:
        return "asciichartpy not installed. Run: pip install asciichartpy"
    if not any(series.values()):
        return "No points to plot."
    parts = []
    for run_name, points in series.items():
        if not points:
            parts.append(f"run: {run_name}\n  (no data)")
            continue
        values = [p.value for p in points]
        steps = [p.step for p in points]
        chart = asciichartpy.plot(values, cfg={"height": height})
        parts.append(
            f"run: {run_name}\n{chart}\n"
            f"  steps {steps[0]}..{steps[-1]}"
        )
    return "\n\n".join(parts)


def render_plot(
    series: Dict[str, List[ScalarPoint]],
    width: int = 80,
    height: int = 24,
    style: str = "plotext",
    metric: str = "value",
    state: Optional[InteractiveState] = None,
) -> str:
    if style == "sparkline":
        return render_sparklines(series)
    if style == "plotext":
        return render_plot_plotext(series, metric=metric, width=width, height=height, state=state)
    if style == "ascii":
        return render_plot_ascii(series, height=max(4, height // 2))
    raise ValueError(f"Unknown plot style: {style!r}. Choose 'plotext', 'sparkline', or 'ascii'.")


def clear_terminal() -> None:
    print("\033[2J\033[H", end="")


def _read_key_nonblocking() -> Optional[str]:
    if not _select.select([sys.stdin], [], [], 0)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        buf = []
        while _select.select([sys.stdin], [], [], 0.05)[0]:
            c = sys.stdin.read(1)
            buf.append(c)
            if c.isalpha() or c == "~":
                break
        return ch + "".join(buf)
    return ch


def _handle_key(key: str, state: InteractiveState, run_names: List[str]) -> bool:
    """Mutates state in-place; returns True if a re-render is needed."""
    if key == "t":
        if state.legend_position != "top":
            state.legend_position = "top"
            return True
    elif key == "b":
        if state.legend_position != "bottom":
            state.legend_position = "bottom"
            return True
    elif key == "\x1b[A":  # up arrow
        if state.cursor_idx > 0:
            state.cursor_idx -= 1
            return True
    elif key == "\x1b[B":  # down arrow
        if run_names and state.cursor_idx < len(run_names) - 1:
            state.cursor_idx += 1
            return True
    elif key == " " and run_names:
        run = run_names[state.cursor_idx]
        state.highlighted_runs ^= {run}
        return True
    elif key == "d":
        state.dark_mode = not state.dark_mode
        return True
    return False


class TBRequestHandler:
    """HTTP request handler for the web UI — mixed in with BaseHTTPRequestHandler at runtime."""

    server_state: dict = {}
    _html_page: str = ""

    def do_GET(self) -> None:
        import urllib.parse

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_html()
        elif path == "/api/runs":
            self._api_runs()
        elif path == "/api/data":
            params = urllib.parse.parse_qs(parsed.query)
            self._api_data(params)
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        body = self.__class__._html_page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: object) -> None:
        import json
        import math

        # json.dumps emits bare ``NaN``/``Infinity`` tokens by default
        # (allow_nan=True), which are invalid JSON and make every browser's
        # strict JSON.parse reject the whole response — surfacing in Safari as
        # "The string did not match the expected pattern". Metrics like
        # pearson_r can legitimately contain NaN for undefined steps, so map
        # any non-finite float to null (Plotly renders it as a gap).
        def _clean(o: object) -> object:
            if isinstance(o, float):
                return o if math.isfinite(o) else None
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            return o

        body = json.dumps(_clean(obj)).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _api_runs(self) -> None:
        state = self.__class__.server_state
        # Wait for background metric probe to finish (blocks this request thread only)
        metrics_ready = state.get("metrics_ready")
        if metrics_ready is not None:
            metrics_ready.wait(timeout=120)
        all_runs = state["all_run_paths"]
        runs_payload = [
            {"id": str(r), "name": r.name, "created": _run_created_time(r)}
            for r in all_runs
        ]
        self._send_json({
            "runs": runs_payload,
            "metrics": state.get("all_metrics", []),
            "logdir": state.get("logdir", ""),
        })

    def _api_data(self, params: dict) -> None:
        import json

        state = self.__class__.server_state
        all_run_paths: List[Path] = state["all_run_paths"]

        run_ids_raw = params.get("runs", [""])[0]
        run_id_set = set(filter(None, run_ids_raw.split(",")))
        requested = [r for r in all_run_paths if str(r) in run_id_set]

        all_metrics: List[str] = state.get("all_metrics", [])
        # Accept multiple metrics via the comma-separated `metrics` param. The
        # single `metric` param stays as a backward-compatible fallback (one
        # metric) so old URLs/bookmarks keep working.
        metrics_raw = params.get("metrics", [None])[0]
        if metrics_raw:
            requested_metrics: List[str] = []
            for m in metrics_raw.split(","):
                m = m.strip()
                if m and m not in requested_metrics:
                    requested_metrics.append(m)
        else:
            single = params.get("metric", [None])[0]
            requested_metrics = [single] if single else (
                [all_metrics[0]] if all_metrics else []
            )

        # Read from the shared cache populated by the background reload thread
        data_lock = state.get("data_lock")
        with data_lock:
            cached = state.get("cached_data", {})
            loaded = {str(r): cached.get(str(r), {}) for r in requested}

        series = []
        for metric in requested_metrics:
            for run_path in requested:
                run_key = str(run_path)
                palette_idx = all_run_paths.index(run_path) % len(_PALETTE_RGB)
                r, g, b = _PALETTE_RGB[palette_idx]
                points = loaded.get(run_key, {}).get(metric, []) if metric else []
                series.append({
                    "run": run_path.name,
                    "run_id": run_key,
                    "color": f"rgb({r},{g},{b})",
                    "metric": metric,
                    "points": [{"step": p.step, "value": p.value, "wall_time": p.wall_time} for p in points],
                })

        self._send_json({"metrics": requested_metrics, "series": series})

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # suppress per-request access log


def run_web_server(args: "argparse.Namespace", logdir: Path) -> int:
    import http.server
    import webbrowser
    import functools

    all_run_paths = discover_runs(logdir)
    if not all_run_paths:
        print(f"No TensorBoard runs found in {logdir}", file=sys.stderr)
        return 1

    all_run_paths = limit_to_most_recent_runs(all_run_paths, args.last)

    if args.runs:
        all_run_paths = filter_runs_by_patterns(all_run_paths, args.runs)
        if not all_run_paths:
            print(
                f"No runs match --runs {args.runs!r} in {logdir}",
                file=sys.stderr,
            )
            return 1

    if args.fast_load:
        loader = functools.partial(load_scalars_fast, stride=args.stride, tail=args.tail)
    else:
        loader = load_scalars

    html_dir = Path(__file__).parent / "web"
    try:
        html_page = (html_dir / "index.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Web UI template not found: {html_dir / 'index.html'}", file=sys.stderr)
        return 1
    html_page = html_page.replace("__REFRESH_DEFAULT__", str(args.refresh))

    metrics_ready = threading.Event()
    data_lock = threading.Lock()

    # Build a concrete handler class by mixing TBRequestHandler with BaseHTTPRequestHandler
    class Handler(TBRequestHandler, http.server.BaseHTTPRequestHandler):
        def handle_one_request(self):
            # A browser that reloads/cancels the page mid-response resets the
            # socket; that surfaces here as ECONNRESET/EPIPE while writing the
            # body. Harmless — swallow it so socketserver doesn't print a
            # scary traceback that reads like a server crash.
            try:
                super().handle_one_request()
            except (ConnectionResetError, BrokenPipeError):
                self.close_connection = True

    Handler.server_state = {
        "all_run_paths": all_run_paths,
        "loader": loader,
        "args": args,
        "logdir": str(logdir),
        "all_metrics": [],
        "cached_data": {},
        "metrics_ready": metrics_ready,
        "data_lock": data_lock,
    }
    Handler._html_page = html_page

    def _bg_reload() -> None:
        """Load data once, signal metrics ready, then keep refreshing on interval.

        Each cycle also re-scans ``logdir`` (via :func:`_discover_new_runs`) so
        runs created after the server started are picked up automatically. Only
        new runs are appended; runs already tracked are never dropped, so a run
        the user is viewing stays put. ``--last`` is intentionally not re-applied
        — a brand-new run is recent by definition and is exactly what
        auto-detection is meant to surface.
        """
        while True:
            new_runs = _discover_new_runs(logdir, args.runs, Handler.server_state["all_run_paths"])

            current_runs = Handler.server_state["all_run_paths"]
            runs_changed = False
            if new_runs:
                # Append (never replace) so existing palette indices stay stable —
                # /api/data keys color off all_run_paths.index(run_path).
                current_runs = [*current_runs, *new_runs]
                with data_lock:
                    Handler.server_state["all_run_paths"] = current_runs
                runs_changed = True

            new_data = loader(current_runs)
            with data_lock:
                Handler.server_state["cached_data"] = new_data
                # Refresh the metric list on the first load and whenever the run
                # set changes, so metrics introduced by a newly appeared run show
                # up in the selector.
                if not metrics_ready.is_set() or runs_changed:
                    Handler.server_state["all_metrics"] = sorted(
                        {m for run_data in new_data.values() for m in run_data}
                    )
            metrics_ready.set()
            time.sleep(max(1.0, args.refresh))

    addr = ("127.0.0.1", args.port)
    try:
        httpd = http.server.ThreadingHTTPServer(addr, Handler)
    except OSError as exc:
        print(f"Cannot start server on port {args.port}: {exc}", file=sys.stderr)
        print(f"Try --port {args.port + 1}", file=sys.stderr)
        return 1

    threading.Thread(target=_bg_reload, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}"
    print(f"tbcli web UI → {url}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _render_latest(
    loaded: Dict[str, Dict[str, List[ScalarPoint]]],
) -> None:
    """Print the latest value of every metric for every loaded run, then exit.

    One block per run; metrics are sorted alphabetically. Runs without any
    scalar data are reported as empty rather than silently skipped.
    """
    for run_key, metrics in loaded.items():
        run_name = Path(run_key).name
        print(f"{_ANSI_BOLD}{run_name}{_ANSI_RESET}")
        if not metrics:
            print("  (no scalar data)")
            print()
            continue
        max_tag = max(len(tag) for tag in metrics)
        for tag in sorted(metrics):
            points = metrics[tag]
            if not points:
                print(f"  {tag:<{max_tag}}  (no data)")
                continue
            last = points[-1]
            values = [p.value for p in points]
            lo, hi = min(values), max(values)
            print(
                f"  {tag:<{max_tag}}  "
                f"last={last.value:.6g}  step={last.step}  "
                f"min={lo:.6g}  max={hi:.6g}  n={len(points)}"
            )
        print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TensorBoard log viewer")
    parser.add_argument("logdir", help="TensorBoard log directory")
    parser.add_argument(
        "--runs",
        help="Comma-separated run selectors, applied in both CLI and --web "
             "mode. Each token is a 1-based run index (CLI only, e.g. '1,3'), "
             "an fnmatch wildcard pattern (e.g. '*exp1*,2024*') matched "
             "against run name or path, or 'all'/'*' for every run. Tokens of "
             "different kinds may be mixed (e.g. '1,3,*exp*'). Only matching "
             "runs are loaded (speeds up startup on large log directories).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        metavar="N",
        help="Load only the N most recent runs (by event-file creation "
             "time). 0 or negative means no limit. Applied to both --web and "
             "CLI/TUI mode, before --runs selection, to speed up loading on "
             "large log directories (default: 0, no limit).",
    )
    parser.add_argument("--metric", help="Metric tag to preselect")
    parser.add_argument("--refresh", type=float, default=5.0, help="Auto-refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Print the latest value of every metric for the selected run(s) "
             "and exit — no plot, no TUI. Useful for a quick glance without "
             "launching the full interface.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--plot-style",
        choices=["plotext", "sparkline", "ascii"],
        default="plotext",
        help="Plot style: 'plotext' (line chart, default), 'sparkline' (compact block chars), or 'ascii' (asciichartpy per-run charts)",
    )
    fast = parser.add_argument_group("fast loading")
    fast.add_argument(
        "--fast-load",
        action="store_true",
        help="Enable fast loading via stride subsampling (avoids parsing most records)",
    )
    fast.add_argument(
        "--stride",
        type=int,
        default=10,
        metavar="S",
        help="With --fast-load: parse 1 in every S records for the history region (default: 10)",
    )
    fast.add_argument(
        "--tail",
        type=int,
        default=500,
        metavar="T",
        help="With --fast-load: number of steps to keep at full fidelity at the end (default: 500)",
    )
    web = parser.add_argument_group("web server")
    web.add_argument(
        "--web",
        action="store_true",
        help="Start a local HTTP server and open the TensorBoard-style web UI in the browser",
    )
    web.add_argument(
        "--port",
        type=int,
        default=6006,
        metavar="PORT",
        help="Port for --web mode (default: 6006)",
    )
    return parser.parse_args(argv)


def _resolve_selected_runs(all_runs: List[Path], run_arg: Optional[str]) -> List[Path]:
    if not all_runs:
        raise RuntimeError("No TensorBoard runs found in the specified directory.")
    if run_arg:
        return resolve_run_selection(all_runs, run_arg)
    if sys.stdin.isatty():
        return prompt_run_selection(all_runs)
    return all_runs


def _resolve_metric(all_metrics: List[str], metric_arg: Optional[str]) -> str:
    if not all_metrics:
        raise RuntimeError("No scalar metrics found in selected runs.")
    if metric_arg:
        if metric_arg not in all_metrics:
            raise RuntimeError(f"Metric '{metric_arg}' not found. Available: {', '.join(all_metrics)}")
        return metric_arg
    if sys.stdin.isatty():
        return prompt_metric_selection(all_metrics)
    return all_metrics[0]


def _summaries_for_metric(
    loaded: Dict[str, Dict[str, List[ScalarPoint]]], metric: str
) -> Dict[str, List[ScalarPoint]]:
    data = {}
    for run, metrics in loaded.items():
        data[Path(run).name] = metrics.get(metric, [])
    return data


def _render_from_loaded(
    loaded: Dict[str, Dict[str, List[ScalarPoint]]],
    metric: str,
    no_plot: bool,
    plot_style: str = "plotext",
    state: Optional[InteractiveState] = None,
) -> None:
    data = _summaries_for_metric(loaded, metric)
    print(f"metric: {metric}")
    if state is not None:
        print("  t=top  b=bottom  ↑/↓=navigate  space=highlight  d=dark/light ctrl+c=quit")
    if not no_plot:
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 80, 24
        print()
        print(render_plot(data, width=cols, height=rows - 10, style=plot_style, metric=metric, state=state))


def _render_once(
    selected_runs: Sequence[Path],
    metric: str,
    no_plot: bool,
    plot_style: str = "plotext",
    state: Optional[InteractiveState] = None,
    loader=None,
) -> None:
    loader = loader or load_scalars
    loaded = loader(selected_runs)
    _render_from_loaded(loaded, metric, no_plot, plot_style, state=state)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logdir = Path(args.logdir).expanduser().resolve()
    if not logdir.exists():
        print(f"Directory does not exist: {logdir}", file=sys.stderr)
        return 2

    if args.web:
        return run_web_server(args, logdir)

    all_runs = discover_runs(logdir)
    all_runs = limit_to_most_recent_runs(all_runs, args.last)
    try:
        selected_runs = _resolve_selected_runs(all_runs, args.runs)
        if args.fast_load:
            import functools
            _loader = functools.partial(load_scalars_fast, stride=args.stride, tail=args.tail)
        else:
            _loader = load_scalars
        loaded = _loader(selected_runs)
        if args.latest:
            _render_latest(loaded)
            return 0
        metric_set = sorted({metric for run_data in loaded.values() for metric in run_data})
        metric = _resolve_metric(metric_set, args.metric)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_names = [Path(r).name for r in selected_runs]
    use_interactive = _HAS_TTY and sys.stdin.isatty() and not args.once

    if use_interactive:
        state        = InteractiveState()
        data_lock    = threading.Lock()
        state_lock   = threading.Lock()
        render_event = threading.Event()
        stop_event   = threading.Event()

        with data_lock:
            latest = loaded  # seed with data already loaded above

        def _bg_load() -> None:
            nonlocal latest
            stop_event.wait(args.refresh)  # first load already done by main thread
            while not stop_event.is_set():
                new_data = _loader(selected_runs)
                with data_lock:
                    latest = new_data
                render_event.set()
                stop_event.wait(args.refresh)

        def _render_loop() -> None:
            while not stop_event.is_set():
                if not render_event.wait(timeout=0.05):
                    continue
                render_event.clear()  # clear before render so mid-render triggers survive
                with data_lock:
                    snapshot = latest
                with state_lock:
                    state_snap = state.copy()
                clear_terminal()
                try:
                    _render_from_loaded(
                        snapshot, metric, args.no_plot, args.plot_style, state=state_snap
                    )
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)

        threading.Thread(target=_bg_load,     daemon=True).start()
        threading.Thread(target=_render_loop,  daemon=True).start()
        render_event.set()  # trigger first paint immediately

        fd = sys.stdin.fileno()
        old_settings = _termios.tcgetattr(fd)
        try:
            _tty.setcbreak(fd)
            while True:  # keyboard only — never touches stdout
                rlist, _, _ = _select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    key = _read_key_nonblocking()
                    if key:
                        with state_lock:
                            changed = _handle_key(key, state, run_names)
                        if changed:
                            render_event.set()
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old_settings)
    else:
        try:
            while True:
                try:
                    clear_terminal()
                    _render_once(selected_runs, metric, args.no_plot, args.plot_style, loader=_loader)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
                if args.once:
                    return 0
                time.sleep(max(0.1, args.refresh))
        except KeyboardInterrupt:
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
