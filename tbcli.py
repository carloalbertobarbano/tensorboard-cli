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


def render_plot(series: Dict[str, List[ScalarPoint]], width: int = 60, height: int = 12) -> str:
    values = []
    for points in series.values():
        values.extend(point.value for point in points)
    if not values:
        return "No points to plot."
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    canvas = [[" " for _ in range(width)] for _ in range(height)]
    for run_index, (run_name, points) in enumerate(series.items()):
        marker = str((run_index + 1) % 10)
        if not points:
            continue
        xs = [point.step for point in points]
        min_x, max_x = min(xs), max(xs)
        x_span = max_x - min_x if max_x != min_x else 1
        for point in points:
            x = int((point.step - min_x) / x_span * (width - 1))
            y = int((point.value - lo) / span * (height - 1))
            y = (height - 1) - y
            canvas[y][x] = marker
    lines = [f"value range [{lo:.4g}, {hi:.4g}]"]
    lines.extend("".join(row) for row in canvas)
    lines.append("legend:")
    for idx, run_name in enumerate(series.keys(), start=1):
        lines.append(f"  {idx % 10}: {run_name}")
    return "\n".join(lines)


def clear_terminal() -> None:
    print("\033[2J\033[H", end="")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TensorBoard log viewer")
    parser.add_argument("logdir", help="TensorBoard log directory")
    parser.add_argument("--runs", help="Comma-separated run indexes or all/*")
    parser.add_argument("--metric", help="Metric tag to preselect")
    parser.add_argument("--refresh", type=float, default=5.0, help="Auto-refresh interval seconds")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    parser.add_argument("--no-plot", action="store_true", help="Disable ASCII plotting")
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


def _render_once(selected_runs: Sequence[Path], metric: str, no_plot: bool) -> None:
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
        print()
        print(render_plot(data))


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
                _render_once(selected_runs, metric, args.no_plot)
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
