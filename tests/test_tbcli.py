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

    def test_render_plot_separate_charts(self):
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
        plot = tbcli.render_plot(series, width=20, height=4)
        self.assertIn("run: run1", plot)
        self.assertIn("run: run2", plot)
        # each run's value should appear as a label above its bar
        self.assertIn("0.1", plot)
        self.assertIn("0.2", plot)

    def test_render_plot_empty_series(self):
        self.assertEqual(tbcli.render_plot({"run1": []}), "No points to plot.")

    def test_render_plot_with_constant_values(self):
        series = {
            "run1": [
                tbcli.ScalarPoint(step=1, value=0.3, wall_time=0),
                tbcli.ScalarPoint(step=2, value=0.3, wall_time=1),
            ]
        }
        plot = tbcli.render_plot(series, width=20, height=4)
        self.assertIn("value range [0.3, 0.3]", plot)
        self.assertIn("run: run1", plot)


if __name__ == "__main__":
    unittest.main()
