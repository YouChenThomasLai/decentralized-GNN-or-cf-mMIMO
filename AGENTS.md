# Repository Guidelines

## Project Structure & Module Organization

Python research code lives under `code/`: `trainer_2.py` trains, `model_2.py` defines the network, `data.py` generates datasets, and `utils_return_indivial_rates.py` provides numerical utilities. `excel_helper.py` summarizes outputs, `plot-v2.py` plots them, and `run_exp-v2.sh` defines parameter sweeps. Research notes and papers belong in `doc/`. Treat `code/results*/`, checkpoints, logs, spreadsheets, and generated plots as artifacts rather than source.

## Build, Test, and Development Commands

Create the documented environment from the repository root:

```bash
conda env create -f code/environment.yml
conda activate decentralized-inference
cd code
```

Run one experiment with `python trainer_2.py --M 2 --N 30 --L 4 --K 8 --pmax_dbm 15 --batch_size 8 --runs 1 --device cuda:0 --out_dir results_local`. Use `--device cpu` when CUDA is unavailable. Run the full parameter sweep with `bash run_exp-v2.sh`. Summarize results with `python excel_helper.py --root results_local`; verify the workbook path in `plot-v2.py` before running it.

## Remote Experiment Workflow

Run GPU sweeps on `lab301-5090` under `~/ThomasLai/code`. Sync only source and configuration files, preserving remote results:

```bash
rsync -av code/{*.py,*.sh,environment.yml,.gitignore} lab301-5090:~/ThomasLai/code/
ssh lab301-5090
cd ~/ThomasLai/code
~/miniforge3/bin/conda env create -f environment.yml
tmux new-session -d -s thomaslai_exp_v2 '~/miniforge3/bin/conda run -n decentralized-inference --no-capture-output ./run_exp-v2.sh 2>&1 | tee run_exp-v2.log'
```

Create the Conda environment only on first setup; if it already exists, verify it instead of recreating it. Before launching, use `tmux has-session -t thomaslai_exp_v2` to avoid duplicate sweeps. Monitor with `tmux attach -t thomaslai_exp_v2` or `tail -f run_exp-v2.log`.

## Coding Style & Naming Conventions

Use Python 3.11, four-space indentation, and PEP 8 conventions. Prefer `snake_case` for functions, variables, and modules; use `PascalCase` for new classes and `UPPER_CASE` for constants. Group standard-library, third-party, and local imports separately. No formatter or linter is configured, so keep changes focused and consistent with nearby code.

## Testing Guidelines

There is no automated test suite or coverage threshold. From `code/`, run `python -m py_compile *.py` and `bash -n run_exp-v2.sh excel_helper.sh`. For numerical changes, run a small experiment and record its command, seed, and key metric; training fixes the seed to `0`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative subjects such as `fix import` and `Fixed the stale -seed-0 input path`. Keep each commit scoped to one logical change. Pull requests should explain the research or implementation impact, list validation commands, and link relevant issues. Include plots or before/after metrics for changes affecting numerical results, and avoid committing caches or incidental generated outputs.
