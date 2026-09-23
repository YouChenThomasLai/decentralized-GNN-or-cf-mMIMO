# Decentralized GNN Beamforming

Research code for joint beamforming in multi-RIS-aided cell-free networks. The
active implementation compares centralized training and decentralized
inference under continuous, 2-bit, and random RIS phase settings.

Research documentation starts at [doc/README.md](doc/README.md). The two main
reports are the [method report](doc/decentralized_ris_methods.md) and the
[experiment report](doc/decentralized_ris_experiments.md).

## Setup

From the repository root:

```bash
conda env create -f code/environment.yml
conda activate decentralized-inference
cd code/decentralized_ris
```

The environment uses Python 3.11 and the CUDA 12.8 PyTorch wheel. CPU execution
is available with `--device cpu`.

## Train and evaluate

Run one seed while exploring a research direction:

```bash
python train.py \
  --M 2 --N 30 --L 4 --K 8 \
  --pmax_dbm 15 --batch_size 8 --n_iter 2000 \
  --test_sample_final 3200 --device cuda:0
```

`M` is the antenna count per AP, `N` is the element count per RIS, `L` is the
RIS count, and `K` is the user count per AP. The default is the `r0` baseline,
with seed `0`. The canonical graph methods use `--arch g0`, `--arch g1`, and
`--arch g2`; their definitions are in the
[method report](doc/decentralized_ris_methods.md#graph-variants). Use
`python train.py --help` for the full interface.

Unassigned exploratory runs default to `artifacts/decentralized_ris/scratch/runs/`, outside the
source tree. Move a retained run into its E01–E09 folder after assigning it to a maintained
experiment. A run has one stable layout:

```text
artifacts/decentralized_ris/scratch/runs/<tag>_iter2000_seed0/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── metrics.npz
└── summary.json
```

Add `--tensorboard` to write TensorBoard events inside the run directory. To
re-evaluate one or more completed runs on the same paired random samples:

```bash
python evaluate.py \
  --runs '../../artifacts/decentralized_ris/scratch/runs/*' \
  --checkpoint best.pt --samples 3200 --device cuda:0
```

The evaluator writes a readable JSON summary and a paired NPZ file for later
statistical comparisons.

## Sweeps and plots

The maintained sweep varies `M` and `pmax_dbm`, always with one seed:

```bash
bash scripts/e08_baseline_sweeps.sh
```

Override `RIS_PYTHON`, `DEVICE`, or `ARTIFACT_ROOT` through environment
variables when running remotely. Plot a completed sweep directly from its JSON
summaries; no Excel conversion step is needed:

```bash
python ../plot-v2.py \
  --results-root ../../artifacts/decentralized_ris/e08_baseline_sweeps/vary_m \
  --x-label '$M$' \
  --output-base ../../artifacts/decentralized_ris/e08_baseline_sweeps/plots/vary_m
```

Each plot command writes PDF and PNG files.

## Code layout

```text
code/decentralized_ris/
├── train.py            # the only training entry point
├── evaluate.py         # paired evaluation, seeding, and checkpoint helpers
├── model.py            # canonical vectorized baseline network
├── variants.py         # R0/R1/G0/G1/G2 action and aggregation variants
├── simulation.py       # topology, channel generation, and model inputs
├── rates.py            # rate objective, phase baselines, and cached evaluation
├── experiments/        # optional analyses, invoked with python -m
├── scripts/            # maintained eNN-prefixed experiment commands
└── tests/              # lightweight numerical regression checks
```

Optional analyses include discrete coordinate descent, a continuous ceiling,
the local-CSI ablation, confidence diagnostics, and discrete-result summaries:

```bash
python -m experiments.discrete_cd --help
python -m experiments.continuous_ceiling --help
python -m experiments.local_csi --help
python -m experiments.confidence --help
python -m experiments.summarize_cd --help
```

Old checkpoints remain loadable through the narrow compatibility logic in
`model.load_checkpoint`. Frozen historical prototypes live under
`code/snapshot_network_scaling/`; do not develop new changes there.

## Development checks

From `code/decentralized_ris/`:

```bash
python -m compileall -q .
python -m tests.test_phase_quantization
python -m tests.test_forward_equivalence
python -m tests.test_rate_equivalence
bash -n scripts/*.sh
```
