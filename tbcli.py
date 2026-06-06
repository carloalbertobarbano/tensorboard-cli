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
) -> Iterator[bytes]:
    """Yield raw event bytes from a TF record file with step-index subsampling.

    Reads all records sequentially (cache-friendly, no seek-syscall overhead).
    For each record outside the tail region the step field is extracted with a
    fast partial decode; the record is kept only when its step index (relative
    to first_step, divided by step_interval) is divisible by stride.

    This makes stride=S mean "keep 1 in S logged steps" regardless of the raw
    step values used by the training loop.

    Records in the last tail_records positions are always yielded (full fidelity).
    """
    tail_start = max(0, total_records - tail_records)
    with open(path, "rb") as fh:
        for i in range(total_records):
            header = fh.read(_TF_RECORD_HEADER_BYTES)
            if len(header) < _TF_RECORD_HEADER_BYTES:
                break
            (data_len,) = struct.unpack("<Q", header[:8])
            data = fh.read(data_len)
            fh.read(_TF_RECORD_FOOTER_BYTES)

            if i >= tail_start:
                yield data
            else:
                step = _parse_event_step(data)
                if step is not None:
                    step_idx = (step - first_step) // step_interval if step_interval > 0 else step
                    if step_idx % stride == 0:
                        yield data


def _find_event_files(run_dir: Path) -> List[str]:
    return sorted(
        str(p) for p in run_dir.iterdir()
        if p.name.startswith(EVENT_FILE_PREFIX)
    )


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
                event_file, stride, tail_records, total, first_step, step_interval
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


def prompt_run_selection(runs: Sequence[Path]) -> List[Path]:
    print("Available runs:")
    for idx, run in enumerate(runs, start=1):
        print(f"  [{idx}] {run}")
    while True:
        raw = input("Select runs (e.g. 1,2 or all/*): ")
        try:
            indexes = parse_selection(raw, len(runs))
            return [runs[i] for i in indexes]
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
        lines.append(f"  {cursor_str}{fmt}{name}  {last_val:.6g}{_ANSI_RESET}")
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

    for i, (run_name, points) in enumerate(series.items()):
        if not points:
            continue
        last_val = points[-1].value
        last_values[run_name] = last_val
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

    legend_str = _render_legend(run_names, last_values, state)
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TensorBoard log viewer")
    parser.add_argument("logdir", help="TensorBoard log directory")
    parser.add_argument("--runs", help="Comma-separated run indexes or all/*")
    parser.add_argument("--metric", help="Metric tag to preselect")
    parser.add_argument("--refresh", type=float, default=5.0, help="Auto-refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
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
    return parser.parse_args(argv)


def _resolve_selected_runs(all_runs: List[Path], run_arg: Optional[str]) -> List[Path]:
    if not all_runs:
        raise RuntimeError("No TensorBoard runs found in the specified directory.")
    if run_arg:
        indexes = parse_selection(run_arg, len(all_runs))
        return [all_runs[i] for i in indexes]
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
        print("  t=top  b=bottom  ↑/↓=navigate  space=highlight  ctrl+c=quit")
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

    all_runs = discover_runs(logdir)
    try:
        selected_runs = _resolve_selected_runs(all_runs, args.runs)
        if args.fast_load:
            import functools
            _loader = functools.partial(load_scalars_fast, stride=args.stride, tail=args.tail)
        else:
            _loader = load_scalars
        loaded = _loader(selected_runs)
        metric_set = sorted({metric for run_data in loaded.values() for metric in run_data})
        metric = _resolve_metric(metric_set, args.metric)
    except RuntimeError as exc:
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
