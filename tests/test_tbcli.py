import io
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

import unittest
from unittest.mock import patch

import tbcli


class FakeAccumulator:
    DATA = {}

    def __init__(self, run_path):
        self.run_path = run_path

    def Reload(self):
        return self

    def Tags(self):
        run_data = self.DATA.get(self.run_path, {})
        return {"scalars": list(run_data.keys())}

    def Scalars(self, tag):
        run_data = self.DATA.get(self.run_path, {})
        return run_data.get(tag, [])


def _event(step, value, wall_time):
    return SimpleNamespace(step=step, value=value, wall_time=wall_time)


class TbCliTests(unittest.TestCase):
    def test_discover_runs_finds_event_directories(self):
        with patch.object(tbcli.os, "walk") as walk:
            walk.return_value = [
                ("/logs/a", [], ["events.out.tfevents.1"]),
                ("/logs/a/sub", [], ["other.txt"]),
                ("/logs/b", [], ["events.out.tfevents.2"]),
            ]
            runs = tbcli.discover_runs(Path("/logs"))
            self.assertEqual(runs, [Path("/logs/a"), Path("/logs/b")])

    def _make_run_with_timestamp(self, tmp: Path, name: str, stamp: int) -> Path:
        run_dir = tmp / name
        run_dir.mkdir(parents=True)
        (run_dir / f"events.out.tfevents.{stamp}.host").write_bytes(b"")
        return run_dir

    def test_limit_to_most_recent_runs_keeps_n_newest(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            oldest = self._make_run_with_timestamp(tmp, "a", 1000)
            mid = self._make_run_with_timestamp(tmp, "b", 2000)
            newest = self._make_run_with_timestamp(tmp, "c", 3000)
            kept = tbcli.limit_to_most_recent_runs([oldest, mid, newest], 2)
            # discovery order preserved, only the 2 newest kept
            self.assertEqual(kept, [mid, newest])

    def test_limit_to_most_recent_runs_no_limit(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            a = self._make_run_with_timestamp(tmp, "a", 1000)
            b = self._make_run_with_timestamp(tmp, "b", 2000)
            runs = [a, b]
            self.assertEqual(tbcli.limit_to_most_recent_runs(runs, 0), runs)
            self.assertEqual(tbcli.limit_to_most_recent_runs(runs, None), runs)
            self.assertEqual(tbcli.limit_to_most_recent_runs(runs, -1), runs)
            # N >= len keeps all
            self.assertEqual(tbcli.limit_to_most_recent_runs(runs, 5), runs)

    def test_parse_selection(self):
        self.assertEqual(tbcli.parse_selection("1,3", 3), [0, 2])
        self.assertEqual(tbcli.parse_selection("all", 3), [0, 1, 2])
        self.assertEqual(tbcli.parse_selection("ALL", 3), [0, 1, 2])
        self.assertEqual(tbcli.parse_selection("*", 3), [0, 1, 2])
        self.assertEqual(tbcli.parse_selection("1,1,2", 3), [0, 1])
        with self.assertRaises(ValueError):
            tbcli.parse_selection("4", 3)
        with self.assertRaises(ValueError):
            tbcli.parse_selection("   ", 3)

    def test_resolve_run_selection_indexes(self):
        runs = [Path("/logs/a"), Path("/logs/b"), Path("/logs/c")]
        self.assertEqual(tbcli.resolve_run_selection(runs, "1,3"),
                         [Path("/logs/a"), Path("/logs/c")])
        self.assertEqual(tbcli.resolve_run_selection(runs, "all"), list(runs))
        self.assertEqual(tbcli.resolve_run_selection(runs, "*"), list(runs))
        # out-of-range index is an error
        with self.assertRaises(ValueError):
            tbcli.resolve_run_selection(runs, "4")
        # empty input is an error
        with self.assertRaises(ValueError):
            tbcli.resolve_run_selection(runs, "   ")

    def test_resolve_run_selection_wildcards(self):
        runs = [Path("/logs/exp1"), Path("/logs/exp2_run"),
                Path("/logs/2024/baseline")]
        # pattern matched against run name
        self.assertEqual(tbcli.resolve_run_selection(runs, "exp1"),
                         [Path("/logs/exp1")])
        # wildcard on name
        self.assertEqual(tbcli.resolve_run_selection(runs, "exp*"),
                         [Path("/logs/exp1"), Path("/logs/exp2_run")])
        # wildcard matched against the full path
        self.assertEqual(tbcli.resolve_run_selection(runs, "*2024*"),
                         [Path("/logs/2024/baseline")])
        # multiple patterns union, duplicates removed, discovery order kept
        self.assertEqual(tbcli.resolve_run_selection(runs, "exp1,*baseline*"),
                         [Path("/logs/exp1"), Path("/logs/2024/baseline")])

    def test_resolve_run_selection_mixed(self):
        runs = [Path("/logs/exp1"), Path("/logs/exp2"), Path("/logs/exp3")]
        # indexes and patterns may be mixed
        self.assertEqual(tbcli.resolve_run_selection(runs, "1,*exp3*"),
                         [Path("/logs/exp1"), Path("/logs/exp3")])

    def test_resolve_run_selection_no_match(self):
        runs = [Path("/logs/exp1")]
        with self.assertRaises(ValueError):
            tbcli.resolve_run_selection(runs, "*nope*")

    def test_discover_new_runs_picks_up_runs_added_after_startup(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_a = self._make_run_with_timestamp(tmp, "a", 1000)
            # Initially only run_a is known to the server.
            self.assertEqual(tbcli._discover_new_runs([tmp], None, [run_a]), [])
            # A new run appears in the logdir after startup.
            run_b = self._make_run_with_timestamp(tmp, "b", 2000)
            new = tbcli._discover_new_runs([tmp], None, [run_a])
            self.assertEqual(new, [run_b])

    def test_discover_new_runs_never_drops_existing_runs(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_a = self._make_run_with_timestamp(tmp, "a", 1000)
            # Even though run_a is the only one on disk, passing a superset of
            # "current" runs (e.g. a since-deleted run) must not return it as new
            # and must not drop anything — existing runs are kept as-is.
            ghost = tmp / "ghost"
            self.assertEqual(tbcli._discover_new_runs([tmp], None, [run_a, ghost]), [])

    def test_discover_new_runs_respects_runs_filter(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            run_a = self._make_run_with_timestamp(tmp, "exp1", 1000)
            # Two new runs appear, only one matches the --runs pattern.
            run_b = self._make_run_with_timestamp(tmp, "exp2", 2000)
            run_c = self._make_run_with_timestamp(tmp, "other", 3000)
            new = tbcli._discover_new_runs([tmp], "exp*", [run_a])
            self.assertEqual(new, [run_b])

    def test_load_scalars_with_loader(self):
        FakeAccumulator.DATA = {
            "/logs/r1": {"loss": [_event(1, 0.5, 0.0), _event(2, 0.25, 1.0)]}
        }
        loaded = tbcli.load_scalars([Path("/logs/r1")], loader_cls=FakeAccumulator)
        self.assertIn("/logs/r1", loaded)
        self.assertIn("loss", loaded["/logs/r1"])
        self.assertEqual(loaded["/logs/r1"]["loss"][-1].value, 0.25)

    # --- sparkline tests ---

    def test_render_sparklines_shows_run_names(self):
        series = {
            "run1": [
                tbcli.ScalarPoint(step=1, value=0.1, wall_time=0),
                tbcli.ScalarPoint(step=2, value=0.2, wall_time=1),
            ],
            "run2": [
                tbcli.ScalarPoint(step=1, value=0.15, wall_time=0),
                tbcli.ScalarPoint(step=2, value=0.18, wall_time=1),
            ],
        }
        out = tbcli.render_sparklines(series)
        self.assertIn("run1", out)
        self.assertIn("run2", out)
        self.assertEqual(len(out.splitlines()), 2)

    def test_render_sparklines_shows_last_value(self):
        series = {
            "run1": [
                tbcli.ScalarPoint(step=10, value=0.5, wall_time=0),
                tbcli.ScalarPoint(step=20, value=0.25, wall_time=1),
            ]
        }
        out = tbcli.render_sparklines(series)
        self.assertIn("last=0.25", out)
        self.assertIn("step=20", out)
        self.assertIn("min=0.25", out)
        self.assertIn("max=0.5", out)

    def test_render_sparklines_uses_block_chars(self):
        series = {
            "run1": [tbcli.ScalarPoint(step=i, value=float(i), wall_time=0) for i in range(8)]
        }
        out = tbcli.render_sparklines(series)
        for ch in "▁▂▃▄▅▆▇█":
            self.assertIn(ch, out)

    def test_render_sparklines_empty_series(self):
        self.assertEqual(tbcli.render_sparklines({"run1": []}), "No points to plot.")

    def test_render_sparklines_no_data_run(self):
        series = {
            "run1": [tbcli.ScalarPoint(step=1, value=0.5, wall_time=0)],
            "run2": [],
        }
        out = tbcli.render_sparklines(series)
        self.assertIn("no data", out)

    def test_render_sparklines_constant_values(self):
        series = {
            "run1": [
                tbcli.ScalarPoint(step=1, value=0.3, wall_time=0),
                tbcli.ScalarPoint(step=2, value=0.3, wall_time=1),
            ]
        }
        out = tbcli.render_sparklines(series)
        self.assertIn("last=0.3", out)

    # --- _render_latest tests ---

    def _capture_latest(self, loaded):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tbcli._render_latest(loaded)
        return buf.getvalue()

    def test_render_latest_shows_all_metrics(self):
        loaded = {
            "/logs/run1": {
                "loss": [tbcli.ScalarPoint(step=10, value=0.5, wall_time=0),
                         tbcli.ScalarPoint(step=20, value=0.25, wall_time=1)],
                "acc": [tbcli.ScalarPoint(step=10, value=0.1, wall_time=0),
                        tbcli.ScalarPoint(step=20, value=0.9, wall_time=1)],
            }
        }
        out = self._capture_latest(loaded)
        self.assertIn("run1", out)
        self.assertIn("loss", out)
        self.assertIn("acc", out)
        # latest values surfaced
        self.assertIn("last=0.25", out)
        self.assertIn("last=0.9", out)
        self.assertIn("step=20", out)

    def test_render_latest_sorts_metrics_and_shows_counts(self):
        loaded = {
            "/logs/run1": {
                "zebra": [tbcli.ScalarPoint(step=1, value=1.0, wall_time=0)],
                "alpha": [tbcli.ScalarPoint(step=1, value=2.0, wall_time=0),
                          tbcli.ScalarPoint(step=2, value=3.0, wall_time=1)],
            }
        }
        out = self._capture_latest(loaded)
        # alphabetical: alpha before zebra
        self.assertLess(out.find("alpha"), out.find("zebra"))
        self.assertIn("n=2", out)
        self.assertIn("n=1", out)

    def test_render_latest_empty_run(self):
        out = self._capture_latest({"/logs/run1": {}})
        self.assertIn("no scalar data", out)

    def test_render_latest_multiple_runs(self):
        loaded = {
            "/logs/a": {"loss": [tbcli.ScalarPoint(step=1, value=0.5, wall_time=0)]},
            "/logs/b": {"loss": [tbcli.ScalarPoint(step=1, value=0.4, wall_time=0)]},
        }
        out = self._capture_latest(loaded)
        self.assertIn("a", out)
        self.assertIn("b", out)

    # --- plotext dispatch tests ---

    def test_render_plot_plotext_returns_string(self):
        series = {
            "run1": [
                tbcli.ScalarPoint(step=1, value=0.5, wall_time=0),
                tbcli.ScalarPoint(step=2, value=0.3, wall_time=1),
            ]
        }
        out = tbcli.render_plot_plotext(series, metric="loss", width=40, height=10)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)

    def test_render_plot_plotext_empty_series(self):
        self.assertEqual(tbcli.render_plot_plotext({"run1": []}), "No points to plot.")

    def test_render_plot_dispatches_sparkline(self):
        series = {"run1": [tbcli.ScalarPoint(step=1, value=0.1, wall_time=0)]}
        out = tbcli.render_plot(series, style="sparkline")
        self.assertIn("run1", out)

    def test_render_plot_dispatches_plotext(self):
        series = {"run1": [tbcli.ScalarPoint(step=1, value=0.1, wall_time=0)]}
        out = tbcli.render_plot(series, style="plotext", width=40, height=10)
        self.assertIsInstance(out, str)

    def test_render_plot_empty_series(self):
        self.assertEqual(tbcli.render_plot({"run1": []}, style="sparkline"), "No points to plot.")
        self.assertEqual(tbcli.render_plot({"run1": []}, style="plotext"), "No points to plot.")

    def test_render_plot_unknown_style_raises(self):
        series = {"run1": [tbcli.ScalarPoint(step=1, value=0.1, wall_time=0)]}
        with self.assertRaises(ValueError):
            tbcli.render_plot(series, style="bogus")

    # --- asciichartpy tests ---

    def test_render_plot_ascii_returns_string(self):
        series = {
            "run1": [
                tbcli.ScalarPoint(step=i, value=float(i) * 0.1, wall_time=0)
                for i in range(1, 6)
            ]
        }
        out = tbcli.render_plot_ascii(series, height=6)
        self.assertIsInstance(out, str)
        self.assertIn("run1", out)
        self.assertTrue(len(out) > 0)

    def test_render_plot_ascii_empty_series(self):
        self.assertEqual(tbcli.render_plot_ascii({"run1": []}), "No points to plot.")

    def test_render_plot_ascii_shows_step_range(self):
        series = {
            "run1": [
                tbcli.ScalarPoint(step=10, value=0.5, wall_time=0),
                tbcli.ScalarPoint(step=20, value=0.8, wall_time=1),
            ]
        }
        out = tbcli.render_plot_ascii(series, height=4)
        self.assertIn("10", out)
        self.assertIn("20", out)

    def test_render_plot_dispatches_ascii(self):
        series = {
            "run1": [tbcli.ScalarPoint(step=i, value=float(i), wall_time=0) for i in range(1, 5)]
        }
        out = tbcli.render_plot(series, style="ascii", height=8)
        self.assertIn("run1", out)

    # --- InteractiveState tests ---

    def test_interactive_state_defaults(self):
        state = tbcli.InteractiveState()
        self.assertEqual(state.legend_position, "top")
        self.assertEqual(state.cursor_idx, 0)
        self.assertEqual(state.highlighted_runs, set())

    def test_interactive_state_highlighted_runs_not_shared(self):
        s1 = tbcli.InteractiveState()
        s2 = tbcli.InteractiveState()
        s1.highlighted_runs.add("x")
        self.assertNotIn("x", s2.highlighted_runs)

    # --- _handle_key tests ---

    def test_handle_key_t_sets_top(self):
        state = tbcli.InteractiveState(legend_position="bottom")
        changed = tbcli._handle_key("t", state, ["run1", "run2"])
        self.assertTrue(changed)
        self.assertEqual(state.legend_position, "top")

    def test_handle_key_b_sets_bottom(self):
        state = tbcli.InteractiveState(legend_position="top")
        changed = tbcli._handle_key("b", state, ["run1", "run2"])
        self.assertTrue(changed)
        self.assertEqual(state.legend_position, "bottom")

    def test_handle_key_t_when_already_top_no_change(self):
        state = tbcli.InteractiveState(legend_position="top")
        self.assertFalse(tbcli._handle_key("t", state, ["run1"]))

    def test_handle_key_b_when_already_bottom_no_change(self):
        state = tbcli.InteractiveState(legend_position="bottom")
        self.assertFalse(tbcli._handle_key("b", state, ["run1"]))

    def test_handle_key_up_arrow_moves_cursor(self):
        state = tbcli.InteractiveState(cursor_idx=2)
        changed = tbcli._handle_key("\x1b[A", state, ["r1", "r2", "r3"])
        self.assertTrue(changed)
        self.assertEqual(state.cursor_idx, 1)

    def test_handle_key_down_arrow_moves_cursor(self):
        state = tbcli.InteractiveState(cursor_idx=0)
        changed = tbcli._handle_key("\x1b[B", state, ["r1", "r2", "r3"])
        self.assertTrue(changed)
        self.assertEqual(state.cursor_idx, 1)

    def test_handle_key_up_at_boundary_no_change(self):
        state = tbcli.InteractiveState(cursor_idx=0)
        self.assertFalse(tbcli._handle_key("\x1b[A", state, ["r1", "r2"]))
        self.assertEqual(state.cursor_idx, 0)

    def test_handle_key_down_at_boundary_no_change(self):
        state = tbcli.InteractiveState(cursor_idx=1)
        self.assertFalse(tbcli._handle_key("\x1b[B", state, ["r1", "r2"]))
        self.assertEqual(state.cursor_idx, 1)

    def test_handle_key_space_highlights_run(self):
        state = tbcli.InteractiveState(cursor_idx=1)
        changed = tbcli._handle_key(" ", state, ["r1", "r2", "r3"])
        self.assertTrue(changed)
        self.assertIn("r2", state.highlighted_runs)

    def test_handle_key_space_toggles_highlight_off(self):
        state = tbcli.InteractiveState(cursor_idx=0, highlighted_runs={"r1"})
        changed = tbcli._handle_key(" ", state, ["r1", "r2"])
        self.assertTrue(changed)
        self.assertNotIn("r1", state.highlighted_runs)

    def test_handle_key_space_multiple_highlights(self):
        state = tbcli.InteractiveState(cursor_idx=0)
        tbcli._handle_key(" ", state, ["r1", "r2"])
        state.cursor_idx = 1
        tbcli._handle_key(" ", state, ["r1", "r2"])
        self.assertEqual(state.highlighted_runs, {"r1", "r2"})

    def test_handle_key_unknown_no_change(self):
        state = tbcli.InteractiveState()
        self.assertFalse(tbcli._handle_key("x", state, ["r1"]))

    # --- _render_legend tests ---

    def test_render_legend_shows_run_names(self):
        state = tbcli.InteractiveState()
        out = tbcli._render_legend(["run1", "run2"], {"run1": 0.5, "run2": 0.3}, state)
        self.assertIn("run1", out)
        self.assertIn("run2", out)

    def test_render_legend_cursor_indicator(self):
        state = tbcli.InteractiveState(cursor_idx=1)
        out = tbcli._render_legend(["r1", "r2"], {"r1": 0.1, "r2": 0.2}, state)
        lines = out.splitlines()
        self.assertNotIn("> ", lines[0])
        self.assertIn("> ", lines[1])

    def test_render_legend_bold_for_highlighted(self):
        state = tbcli.InteractiveState(highlighted_runs={"run1"})
        out = tbcli._render_legend(["run1", "run2"], {"run1": 0.5, "run2": 0.3}, state)
        lines = out.splitlines()
        self.assertIn("\033[1m", lines[0])

    def test_render_legend_dim_for_non_highlighted(self):
        state = tbcli.InteractiveState(highlighted_runs={"run1"})
        out = tbcli._render_legend(["run1", "run2"], {"run1": 0.5, "run2": 0.3}, state)
        lines = out.splitlines()
        self.assertIn("\033[2m", lines[1])

    def test_render_legend_no_bold_dim_when_no_highlight(self):
        state = tbcli.InteractiveState()
        out = tbcli._render_legend(["run1", "run2"], {"run1": 0.5, "run2": 0.3}, state)
        self.assertNotIn("\033[1m", out)
        self.assertNotIn("\033[2m", out)

    def test_render_legend_last_value_shown(self):
        state = tbcli.InteractiveState()
        out = tbcli._render_legend(["run1"], {"run1": 1.23456}, state)
        self.assertIn("1.23456", out)

    # --- render_plot_plotext with state tests ---

    def test_render_plot_plotext_with_state_returns_string(self):
        series = {
            "run1": [tbcli.ScalarPoint(step=1, value=0.5, wall_time=0)],
            "run2": [tbcli.ScalarPoint(step=1, value=0.3, wall_time=0)],
        }
        state = tbcli.InteractiveState()
        out = tbcli.render_plot_plotext(series, metric="loss", width=40, height=10, state=state)
        self.assertIsInstance(out, str)
        self.assertIn("run1", out)
        self.assertIn("run2", out)

    def test_render_plot_plotext_with_state_legend_bottom(self):
        series = {"run1": [tbcli.ScalarPoint(step=1, value=0.5, wall_time=0)]}
        state = tbcli.InteractiveState(legend_position="bottom")
        out = tbcli.render_plot_plotext(series, metric="loss", width=40, height=10, state=state)
        self.assertIsInstance(out, str)
        self.assertIn("run1", out)
        # legend (last line block) comes after the plot body
        legend_pos = out.rfind("run1")
        self.assertGreater(legend_pos, 0)

    def test_render_plot_plotext_with_state_top_vs_bottom_order(self):
        series = {"run1": [tbcli.ScalarPoint(step=1, value=0.5, wall_time=0)]}
        state_top = tbcli.InteractiveState(legend_position="top")
        state_bot = tbcli.InteractiveState(legend_position="bottom")
        top_out = tbcli.render_plot_plotext(series, width=40, height=10, state=state_top)
        bot_out = tbcli.render_plot_plotext(series, width=40, height=10, state=state_bot)
        # In top mode, run name appears earlier (legend first)
        top_pos = top_out.find("run1")
        bot_pos = bot_out.find("run1")
        self.assertLess(top_pos, bot_pos)

    def test_render_plot_with_state_passes_through(self):
        series = {"run1": [tbcli.ScalarPoint(step=1, value=0.1, wall_time=0)]}
        state = tbcli.InteractiveState()
        out = tbcli.render_plot(series, style="plotext", width=40, height=10, state=state)
        self.assertIsInstance(out, str)

    def test_render_plot_plotext_with_highlight_returns_string(self):
        series = {
            "run1": [tbcli.ScalarPoint(step=i, value=float(i) * 0.1, wall_time=0) for i in range(5)],
            "run2": [tbcli.ScalarPoint(step=i, value=float(i) * 0.2, wall_time=0) for i in range(5)],
        }
        state = tbcli.InteractiveState(highlighted_runs={"run1"})
        out = tbcli.render_plot_plotext(series, width=60, height=12, state=state)
        self.assertIsInstance(out, str)
        self.assertIn("run1", out)
        self.assertIn("run2", out)


def _write_fake_tfrecord(fh: io.RawIOBase, data: bytes) -> None:
    """Write a TF record with zero CRCs (sufficient for readers that skip CRC validation)."""
    fh.write(struct.pack("<Q", len(data)))  # uint64 length
    fh.write(struct.pack("<I", 0))          # masked CRC of length (fake)
    fh.write(data)                          # payload
    fh.write(struct.pack("<I", 0))          # masked CRC of payload (fake)


def _make_scalar_event(step: int, tag: str, value: float, wall_time: float = 1.0) -> bytes:
    """Return serialized Event proto with one scalar summary value."""
    from tensorboard.compat.proto.event_pb2 import Event
    ev = Event()
    ev.wall_time = wall_time
    ev.step = step
    sv = ev.summary.value.add()
    sv.tag = tag
    sv.simple_value = value
    return ev.SerializeToString()


def _write_fake_events_file(path: str, steps_and_tags: list) -> None:
    """Write a minimal TF events file.

    steps_and_tags: list of (step, tag, value) triples.
    """
    with open(path, "wb") as fh:
        for step, tag, value in steps_and_tags:
            _write_fake_tfrecord(fh, _make_scalar_event(step, tag, value))


class FastLoadTests(unittest.TestCase):
    # --- _parse_event_step ---

    def test_parse_event_step_normal_event(self):
        from tensorboard.compat.proto.event_pb2 import Event
        ev = Event()
        ev.wall_time = 1.0
        ev.step = 12345
        sv = ev.summary.value.add()
        sv.tag = "train/loss"
        sv.simple_value = 0.5
        self.assertEqual(tbcli._parse_event_step(ev.SerializeToString()), 12345)

    def test_parse_event_step_large_step(self):
        from tensorboard.compat.proto.event_pb2 import Event
        ev = Event()
        ev.wall_time = 1.0
        ev.step = 100_000
        self.assertEqual(tbcli._parse_event_step(ev.SerializeToString()), 100_000)

    def test_parse_event_step_no_step_field(self):
        # wall_time=1.0 but no explicit step → step field omitted (proto3 default=0)
        from tensorboard.compat.proto.event_pb2 import Event
        ev = Event()
        ev.wall_time = 1.0
        # step not set → proto3 omits it from serialized bytes
        self.assertEqual(tbcli._parse_event_step(ev.SerializeToString()), 0)

    def test_parse_event_step_empty_bytes(self):
        self.assertIsNone(tbcli._parse_event_step(b""))

    def test_parse_event_step_invalid_bytes(self):
        self.assertIsNone(tbcli._parse_event_step(b"\x00" * 20))

    # --- load_scalars_fast end-to-end ---

    def _make_run_dir(self, steps_and_tags):
        """Create a temporary run directory with a fake events file."""
        td = tempfile.mkdtemp()
        events_path = Path(td) / "events.out.tfevents.0000.host.0.0"
        _write_fake_events_file(str(events_path), steps_and_tags)
        return Path(td)

    def test_load_scalars_fast_all_tags_present(self):
        # 3 steps × 2 tags, stride=1 → all points returned
        run = self._make_run_dir([
            (10, "loss", 1.0), (10, "lr", 0.1),
            (20, "loss", 0.8), (20, "lr", 0.09),
            (30, "loss", 0.6), (30, "lr", 0.08),
        ])
        data = tbcli.load_scalars_fast([run], stride=1, tail=0)
        self.assertIn("loss", data[str(run)])
        self.assertIn("lr", data[str(run)])
        self.assertEqual(len(data[str(run)]["loss"]), 3)
        self.assertEqual(len(data[str(run)]["lr"]), 3)

    def test_load_scalars_fast_stride_keeps_every_nth_step(self):
        # 10 steps, stride=2 → every other step (indices 0, 2, 4, 6, 8)
        steps_tags = [(s * 10, "loss", float(s)) for s in range(1, 11)]
        run = self._make_run_dir(steps_tags)
        data = tbcli.load_scalars_fast([run], stride=2, tail=0)
        pts = data[str(run)]["loss"]
        # stride=2 keeps step_index=0,2,4,6,8 → 5 points
        self.assertEqual(len(pts), 5)

    def test_load_scalars_fast_tail_always_kept(self):
        # 10 steps, stride=10 keeps only step_index=0, but tail=3 keeps last 3
        steps_tags = [(s * 10, "loss", float(s)) for s in range(1, 11)]
        run = self._make_run_dir(steps_tags)
        data = tbcli.load_scalars_fast([run], stride=10, tail=3)
        pts = data[str(run)]["loss"]
        last_step = pts[-1].step
        self.assertEqual(last_step, 100)  # tail contains the last step

    def test_load_scalars_fast_all_tags_for_kept_steps(self):
        # 5 steps × 2 tags; stride=2 should keep same steps for both tags
        steps_tags = []
        for s in range(1, 6):
            steps_tags.append((s * 5, "loss", float(s)))
            steps_tags.append((s * 5, "acc", float(s) * 0.1))
        run = self._make_run_dir(steps_tags)
        data = tbcli.load_scalars_fast([run], stride=2, tail=0)
        loss_steps = {p.step for p in data[str(run)]["loss"]}
        acc_steps  = {p.step for p in data[str(run)]["acc"]}
        # Both metrics should cover the same steps
        self.assertEqual(loss_steps, acc_steps)

    def test_load_scalars_fast_empty_run(self):
        td = tempfile.mkdtemp()
        events_path = Path(td) / "events.out.tfevents.0000.host.0.0"
        with open(str(events_path), "wb"):
            pass  # empty file
        data = tbcli.load_scalars_fast([Path(td)], stride=10, tail=500)
        self.assertEqual(data[str(Path(td))], {})

    def test_send_json_sanitizes_non_finite_floats(self):
        # Metrics like pearson_r can contain NaN/Infinity for undefined
        # steps. json.dumps emits bare ``NaN``/``Infinity`` tokens by default,
        # which are invalid JSON and make the browser's strict JSON.parse
        # reject the whole response. _send_json must map them to null.
        import json

        class _FakeWFile:
            def __init__(self):
                self.buf = bytearray()
            def write(self, b):
                self.buf += b

        class _FakeHandler:
            def __init__(self):
                self.wfile = _FakeWFile()
            def send_response(self, code): pass
            def send_header(self, k, v): pass
            def end_headers(self): pass

        fh = _FakeHandler()
        payload = {
            "metric": "eval/pc2/pearson_r",
            "series": [{
                "run": "a",
                "points": [
                    {"step": 1, "value": float("nan"), "wall_time": 1.0},
                    {"step": 2, "value": float("inf"), "wall_time": 2.0},
                    {"step": 3, "value": float("-inf"), "wall_time": 3.0},
                    {"step": 4, "value": 0.5, "wall_time": 4.0},
                ],
            }],
        }
        tbcli.TBRequestHandler._send_json(fh, payload)
        body = bytes(fh.wfile.buf)
        self.assertNotIn(b"NaN", body)
        self.assertNotIn(b"Infinity", body)
        # strict parse must succeed, like a browser: reject bare NaN/Infinity
        def _reject(const):
            raise ValueError(f"bare JSON constant {const!r}")
        decoded = json.loads(body.decode("utf-8"), parse_constant=_reject)
        vals = decoded["series"][0]["points"]
        self.assertIsNone(vals[0]["value"])
        self.assertIsNone(vals[1]["value"])
        self.assertIsNone(vals[2]["value"])
        self.assertEqual(vals[3]["value"], 0.5)


def _envarint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _encode_scalar_event(step: int, tag: str, value: float = 0.0, wall_time: float = 1.0) -> bytes:
    """Hand-encode a minimal scalar Event proto without depending on tensorboard.

    Matches the layout _parse_event_step and tensorboard's Event parser expect:
      field 1 wall_time (fixed64) | field 2 step (varint) | field 5 summary
    with Summary.Value: field 1 tag (string) | field 2 simple_value (fixed32).
    """
    tag_bytes = tag.encode("utf-8")
    value_msg = b"\x0a" + _envarint(len(tag_bytes)) + tag_bytes + b"\x15" + struct.pack("<f", value)
    summary_msg = b"\x0a" + _envarint(len(value_msg)) + value_msg
    return (
        b"\x09" + struct.pack("<d", wall_time)
        + b"\x10" + _envarint(step)
        + b"\x2a" + _envarint(len(summary_msg)) + summary_msg
    )


def _rd_varint(data: bytes, pos: int):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            return result, pos


def _decode_value(data: bytes):
    pos, tag, value = 0, None, None
    while pos < len(data):
        key, pos = _rd_varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            _, pos = _rd_varint(data, pos)
        elif wire == 2:
            ln, pos = _rd_varint(data, pos)
            if field == 1:
                tag = data[pos:pos + ln].decode("utf-8")
            pos += ln
        elif wire == 5:
            if field == 2:
                value = struct.unpack("<f", data[pos:pos + 4])[0]
            pos += 4
        elif wire == 1:
            pos += 8
    return tag, value


def _decode_summary(data: bytes):
    pos, tag, value = 0, None, None
    while pos < len(data):
        key, pos = _rd_varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == 2:
            ln, pos = _rd_varint(data, pos)
            if field == 1:
                tag, value = _decode_value(data[pos:pos + ln])
            pos += ln
        else:
            break
    return tag, value


def _decode_event(data: bytes):
    """Inverse of _encode_scalar_event -> (step, tag, value). No tensorboard needed."""
    pos, step, tag, value = 0, 0, None, None
    while pos < len(data):
        key, pos = _rd_varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, pos = _rd_varint(data, pos)
            if field == 2:
                step = v
        elif wire == 1:
            pos += 8
        elif wire == 5:
            pos += 4
        elif wire == 2:
            ln, pos = _rd_varint(data, pos)
            if field == 5:
                tag, value = _decode_summary(data[pos:pos + ln])
            pos += ln
    return step, tag, value


def _write_hand_events_file(path: str, records) -> None:
    """Write a TF events file from (step, tag, value) triples without tensorboard."""
    with open(path, "wb") as fh:
        for step, tag, value in records:
            _write_fake_tfrecord(fh, _encode_scalar_event(step, tag, float(value)))


class CadenceFastLoadTests(unittest.TestCase):
    """Regression tests for the cadence-aware stride subsampling.

    These run without tensorboard installed: they feed hand-encoded TFRecord
    bytes straight into _iter_tfrecord_stride_by_step, which is the function
    the cadence bug lived in.
    """

    def _make_file(self, records):
        td = tempfile.mkdtemp()
        path = Path(td) / "events.out.tfevents.0000.host.0.0"
        _write_hand_events_file(str(path), records)
        return str(path)

    def test_sparse_cadence_metric_keeps_full_history(self):
        # Dense metrics (loss, acc, lr) are logged every step -> 3 records/step.
        # A sparse metric "sp" is logged only at steps 20, 50, 80, so those steps
        # carry 4 records instead of the dense norm of 3. With stride=3 a naive
        # step-index % stride filter would drop steps 20/50/80 (indices 2/5/8, none
        # divisible by 3), erasing all of "sp"'s history. The cadence-aware loader
        # must keep those atypical steps in full.
        records = []
        for i in range(10):
            s = i * 10
            records.append((s, "loss", float(i)))
            records.append((s, "acc", float(i) * 0.1))
            records.append((s, "lr", float(i) * 0.01))
        for s, v in [(20, 1.0), (50, 2.0), (80, 3.0)]:
            records.append((s, "sp", float(v)))
        path = self._make_file(records)
        total = tbcli._count_tfrecord_events(path)

        yielded = list(tbcli._iter_tfrecord_stride_by_step(
            path, stride=3, tail_records=0, total_records=total,
            first_step=0, step_interval=10, records_per_step=3,
        ))

        by_tag = {}
        for raw in yielded:
            step, tag, _ = _decode_event(raw)
            by_tag.setdefault(tag, []).append(step)

        # Dense metrics: decimated to step indices 0, 3, 6, 9 -> steps 0,30,60,90.
        self.assertEqual(sorted(set(by_tag["loss"])), [0, 30, 60, 90])
        self.assertEqual(sorted(set(by_tag["acc"])), [0, 30, 60, 90])
        self.assertEqual(sorted(set(by_tag["lr"])), [0, 30, 60, 90])
        # Sparse metric: ALL of its history survives the decimation.
        self.assertEqual(sorted(by_tag["sp"]), [20, 50, 80])

    def test_dense_only_run_decimates_and_keeps_tail(self):
        # Sanity: with no sparse metric, stride decimates evenly and the tail is
        # kept at full fidelity.
        records = [(i * 10, "loss", float(i)) for i in range(10)]
        path = self._make_file(records)
        total = tbcli._count_tfrecord_events(path)

        yielded = list(tbcli._iter_tfrecord_stride_by_step(
            path, stride=2, tail_records=2, total_records=total,
            first_step=0, step_interval=10, records_per_step=1,
        ))
        steps = sorted({tbcli._parse_event_step(r) for r in yielded})
        # stride=2 keeps indices 0,2,4,6,8 -> steps 0,20,40,60,80;
        # tail_records=2 keeps the last two records -> steps 80,90.
        self.assertEqual(steps, [0, 20, 40, 60, 80, 90])


if __name__ == "__main__":
    unittest.main()
