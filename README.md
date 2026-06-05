# tensorboard-cli

Lightweight CLI interface to inspect TensorBoard logs from terminals, including remote clusters without web access.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python tbcli.py /path/to/logdir
```

The CLI will:
- discover all runs under the provided folder
- prompt for run selection
- prompt for scalar metric selection
- print latest scalar values per run
- render a minimal ASCII plot (disable with `--no-plot`)
- auto-refresh from disk every few seconds

Useful flags:

```bash
python tbcli.py /path/to/logdir --runs all --metric loss --refresh 2
python tbcli.py /path/to/logdir --once --no-plot
```
