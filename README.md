# Decentralized GNN Beamforming

Research code for joint beamforming in multi-RIS-aided cell-free networks. The
training pipeline compares centralized and decentralized GNN inference with
continuous, 2-bit discrete, and random RIS phase shifts.

## Setup

Create the Conda environment from the repository root:

```bash
conda env create -f code/environment.yml
conda activate decentralized-inference
cd code/stage0
```

The environment uses Python 3.11 and installs PyTorch from the CUDA 12.8 wheel
index. Training also supports CPU execution with `--device cpu`.

## Run an experiment

```bash
python trainer_2.py \
  --M 2 --N 30 --L 4 --K 8 \
  --pmax_dbm 15 --batch_size 8 --runs 1 \
  --device cuda:0 --out_dir results_local
```

Here, `M` is the number of antennas per access point, `N` the number of RIS
elements, `L` the number of RISs, `K` the number of users, and `pmax_dbm` the
per-access-point transmit-power limit in dBm. Run `python trainer_2.py --help`
for all options.

The random seed is fixed to `0`. Results are written under a parameter-named
directory such as:

```text
results_local/M2_N30_L4_K8_P15.0/run0/
├── arrays/
├── final_eval/
├── logs/
└── models/
```

Inspect training logs with:

```bash
tensorboard --logdir results_local
```

## Run and summarize sweeps

`run_exp-v2.sh` runs the configured `M` and `pmax_dbm` sweeps on `cuda:0`:

```bash
bash run_exp-v2.sh
```

Summarize the default `M` sweep with the wrapper script:

```bash
bash excel_helper.sh
```

Alternatively, pass either result directory directly to the Python helper:

```bash
python excel_helper.py --root results_batch_8_BS-radius_200_RIS-radius_100_vary_M
python excel_helper.py --root results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
```

Plot either summary by passing the workbook, x-axis label, and output path
without a file extension:

```bash
python ../plot-v2.py \
  --excel results_batch_8_BS-radius_200_RIS-radius_100_vary_M/summary_M.xlsx \
  --x-label '$M$' \
  --output-base stage0_ppt_assets/stage0_vary_M

python ../plot-v2.py \
  --excel results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax/summary_P.xlsx \
  --x-label '$P_{\mathrm{max}}$ (dBm)' \
  --output-base stage0_ppt_assets/stage0_vary_Pmax
```

Each command writes both PDF and PNG versions of the plot.

## Code guide

### `stage0/trainer_2.py`

The main training and inference module. `Trainer` creates `MyDataLoader`,
associates every access point (AP) with the RISs, builds the `node_update`
network, and manages training, evaluation, logging, and saved artifacts.

- `train_batch()` performs one centralized training step and returns loss, sum
  rate, and per-user rates.
- `train()` runs the fixed 2,000-iteration training loop, periodic validation,
  final evaluation, and artifact saving.
- `eval()` compares centralized and decentralized inference using continuous,
  2-bit discrete, random, and random-discrete RIS phases.

### `stage0/model_2.py`

Defines the GNN and readout networks:

- `initial_layer` encodes user and RIS features.
- `node_update_layer` performs message passing and updates node features.
- `BS_readout` predicts beamforming weights.
- `RIS_readout_AP` produces an RIS prediction from each AP, and `RIS_merge`
  combines those predictions into normalized phase vectors.
- `coeff_DNN2` predicts each AP's power-control coefficient.
- `node_update` connects the full model. Its `training=True` forward path is
  centralized; `training=False` uses decentralized AP-local inputs.

### `stage0/data.py`

Defines the wireless topology and data pipeline. `Base_station` and `RIS` hold
topology and channel state. `MyDataLoader` generates channels, associates users
with APs using received signal strength, creates centralized or decentralized
model inputs, and delegates sum-rate loss calculation to
`utils_return_indivial_rates.py`.

The key methods are `BS_RIS_association()`, `BS_user_association()`,
`load_data()`, `gen_training_data()`, `gen_testing_data()`, and
`compute_loss()`.

### `stage0/utils_return_indivial_rates.py`

Contains geometry, channel simulation, RIS quantization, and rate utilities.
`Channel` generates small- and large-scale fading; `generate_channel()` builds
the AP-RIS-user and direct AP-user channels; `discrete_mapping()` maps phases to
a requested bit resolution; and `cal_loss()` computes user rates, sum rate, and
the negative-sum-rate training loss.

### `stage0/excel_helper.py`

Scans parameter-named experiment directories, detects the swept variable, and
writes the `run0` final-evaluation metrics to `summary_<variable>.xlsx`.

### `plot-v2.py`

Reads a summary workbook and produces publication-style PDF and PNG plots
comparing the selected centralized and decentralized methods. Pass the input
workbook, x-axis label, and output path with `--excel`, `--x-label`, and
`--output-base`.

The Stage 0 parameter sweeps are defined in `stage0/run_exp-v2.sh`. Stage 1
and Stage 2 keep their own source, scripts, and local results under
`stage1/` and `stage2/`; research notes and reference material are under
`doc/`.

## Development checks

From `code/stage0/`:

```bash
python -m py_compile *.py
bash -n run_exp-v2.sh excel_helper.sh
python test_discrete_mapping.py
```
