#!/usr/bin/env python3
import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


EVENT_FILE_PREFIX = "events.out.tfevents."


@dataclass
class ScalarPoint:
    step: int
    value: float
    wall_time: float


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


def render_plot_plotext(
    series: Dict[str, List[ScalarPoint]],
    metric: str = "value",
    width: int = 80,
    height: int = 24,
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
    for run_name, points in series.items():
        if not points:
            continue
        plt.plot([p.step for p in points], [p.value for p in points], label=run_name)
    return plt.build()


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
) -> str:
    if style == "sparkline":
        return render_sparklines(series)
    if style == "plotext":
        return render_plot_plotext(series, metric=metric, width=width, height=height)
    if style == "ascii":
        return render_plot_ascii(series, height=max(4, height // 2))
    raise ValueError(f"Unknown plot style: {style!r}. Choose 'plotext', 'sparkline', or 'ascii'.")


def clear_terminal() -> None:
    print("\033[2J\033[H", end="")


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


def _render_once(
    selected_runs: Sequence[Path],
    metric: str,
    no_plot: bool,
    plot_style: str = "plotext",
) -> None:
    loaded = load_scalars(selected_runs)
    data = _summaries_for_metric(loaded, metric)
    print(f"metric: {metric}")
    for run_name, points in data.items():
        if points:
            last = points[-1]
            print(f"  {run_name}: step={last.step} value={last.value:.6g}")
        else:
            print(f"  {run_name}: no data")
    if not no_plot:
        try:
            cols, rows = os.get_terminal_size()
        except OSError:
            cols, rows = 80, 24
        print()
        print(render_plot(data, width=cols, height=rows - 10, style=plot_style, metric=metric))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logdir = Path(args.logdir).expanduser().resolve()
    if not logdir.exists():
        print(f"Directory does not exist: {logdir}", file=sys.stderr)
        return 2

    all_runs = discover_runs(logdir)
    try:
        selected_runs = _resolve_selected_runs(all_runs, args.runs)
        loaded = load_scalars(selected_runs)
        metric_set = sorted({metric for run_data in loaded.values() for metric in run_data})
        metric = _resolve_metric(metric_set, args.metric)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        while True:
            try:
                clear_terminal()
                _render_once(selected_runs, metric, args.no_plot, args.plot_style)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if args.once:
                return 0
            time.sleep(max(0.1, args.refresh))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
