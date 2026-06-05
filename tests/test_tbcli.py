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


if __name__ == "__main__":
    unittest.main()
