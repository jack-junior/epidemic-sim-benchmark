# Experiments

Run experiment commands from the repository root with `uv run`.

## ECS vs Mesa benchmark

The benchmark runner provides epidemiological parity and system profiling:

```bash
# Compare outcome distributions over independent replicates
uv run python -m experiments.benchmark_runner parity --replicates 100 --n-agents 300

# Profile a fixed number of simulation steps
uv run python -m experiments.benchmark_runner profile --n-agents 1000 --steps 50
```

Parity results are printed as a table and written to `results/`. The profiling
command reports per-system timings for both implementations. Repeat profiling
runs before drawing conclusions from timings.

The underlying commands can also be run independently:

```bash
uv run python -m experiments.parity.compare_implementations --help
uv run python -m experiments.profiling.profile_models --help
```

## Quick run

Run a small parity experiment using the example shell script:

```bash
bash experiments/scripts/quick_run.sh
```
