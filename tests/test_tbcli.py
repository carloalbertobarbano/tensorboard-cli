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


if __name__ == "__main__":
    unittest.main()
