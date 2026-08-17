# tensorboard-cli

A lightweight TensorBoard log viewer that works **two ways**:

- **CLI** — inspect runs and scalar metrics right in the terminal, with live plots. Ideal for remote clusters and machines without a browser.
- **Web UI** — a local, TensorBoard-style web interface rendered with Plotly, for when you want an interactive chart.

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

By default the CLI will:

- discover all runs under the provided folder
- prompt for run selection
- prompt for scalar metric selection
- print the latest scalar values per run (step + value)
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

### Fast loading

For large log directories, `--fast-load` parses only 1 in every `--stride` records for the history region while keeping the last `--tail` steps at full fidelity — much faster startup without losing the recent detail:

```bash
uv run tbcli.py /path/to/logdir --fast-load --stride 10 --tail 500
```

## Web UI

Add `--web` to start a local HTTP server and open a TensorBoard-style interface in your browser:

```bash
uv run tbcli.py /path/to/logdir --web
```

This serves the UI on `http://127.0.0.1:6006` and opens it automatically. Pick a different port with `--port`:

```bash
uv run tbcli.py /path/to/logdir --web --port 8080
```

In `--web` mode the `--runs` flag takes comma-separated wildcard patterns (fnmatch, e.g. `*exp1*,2024*`) matched against run name or path — only matching runs are loaded, which speeds up startup on large log directories:

```bash
uv run tbcli.py /path/to/logdir --web --runs '*exp1*,2024*'
```

The web UI refreshes from disk in the background on the `--refresh` interval (default 5s), so new points show up live without reloading the page.

### Web UI flags

| Flag | Default | Description |
|---|---|---|
| `--web` | off | Start the local web server and open the UI in a browser |
| `--port PORT` | `6006` | Port for the web server |
| `--runs PATTERNS` | all | Comma-separated wildcard patterns to filter runs |
| `--metric TAG` | — | Metric tag to preselect |
| `--refresh SECONDS` | `5` | Background refresh interval |
| `--fast-load` | off | Enable stride subsampling for faster startup |