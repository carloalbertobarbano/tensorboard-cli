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


if __name__ == "__main__":
    unittest.main()
