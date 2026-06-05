# tensorboard-cli

Lightweight CLI interface to inspect TensorBoard logs from terminals, including remote clusters without web access.

## Install

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Usage

```bash
uv run tbcli.py /path/to/logdir
```

The CLI will:
- discover all runs under the provided folder
- prompt for run selection
- prompt for scalar metric selection
- print latest scalar values per run (step + value for each run)
- render a terminal plot (disable with `--no-plot`)
- auto-refresh from disk every few seconds

### Plot styles

Three styles are available via `--plot-style`:

| Style | Flag | Description |
|---|---|---|
| `plotext` | _(default)_ | Smooth Unicode line chart with labeled axes, all runs overlaid |
| `sparkline` | `--plot-style sparkline` | One compact line per run using block chars (`▁▂▃▄▅▆▇█`), with last/min/max values |
| `ascii` | `--plot-style ascii` | Classic ASCII line chart with Y-axis value labels, one chart per run |

### Flags

```bash
# preselect runs and metric, faster refresh
uv run tbcli.py /path/to/logdir --runs all --metric loss --refresh 2

# render once and exit, no plot
uv run tbcli.py /path/to/logdir --once --no-plot

# choose a plot style
uv run tbcli.py /path/to/logdir --plot-style sparkline
uv run tbcli.py /path/to/logdir --plot-style ascii
uv run tbcli.py /path/to/logdir --plot-style plotext
```
