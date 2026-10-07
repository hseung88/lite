# Derivative Gaussian Processes on a Two-Direction Budget

## Installation

Use Python 3.10 or newer. Install a CUDA-enabled PyTorch build for GPU runs.
Run the following commands from the repository root:

```bash
python -m venv lite
source lite/bin/activate
python -m pip install -e .
```

## Command examples

Common flags apply to the selected methods listed below. A method-specific flag
such as `--tera-train-batch-size` overrides the corresponding common flag,
regardless of argument order. Existing method-specific flags and flat YAML
configs are also supported. Use `--help` to list all options.

| Common options | MD22 | BO |
| --- | --- | --- |
| `--m` | LITE, TERA variants, Vecchia | LITE, TERA variants |
| `--lr`, `--train-epochs` | Standard GP, LITE, TERA variants, Vecchia, DSoftKI, DDSVGP | `--lr`: LITE, TERA variants |
| `--train-steps` | Standard GP, LITE, TERA variants, Vecchia | LITE, TERA variants |
| `--train-batch-size`, `--prediction-batch-size` | LITE, TERA variants, Vecchia, DSoftKI, DDSVGP | LITE, TERA variants |
| `--initial-train-steps`, `--update-train-steps`, `--refit-every` | — | LITE, TERA variants |
| `--learn-lengthscale`, `--learn-outputscale`, `--learn-sigma-f` | Standard GP, LITE, TERA variants, Vecchia | LITE, TERA variants |
| `--learn-sigma-g`, `--gradient-noise-model` | LITE, TERA variants | LITE, TERA variants |

Sequential `tera` predicts one target at a time; prediction batch size controls
`tera_batched`. VBO and TuRBO retain their own training flags, for example
`--vbo-fit-maxiter` and `--turbo-gp-train-steps`. BO's `--query-batch-size`
(alias `--batch-size`) counts objective evaluations per iteration.

### MD22 regression

Download the six MD22 datasets into `data/md22`:

```bash
mkdir -p data/md22
for dataset in DHA AT-AT stachyose AT-AT-CG-CG buckyball-catcher double-walled_nanotube; do
  wget -nc -P data/md22 \
    "https://www.quantum-machine.org/gdml/repo/datasets/md22_${dataset}.npz"
done
```

Run the main LITE and batched TERA comparison with `m=30`, one training epoch
and training/prediction batches of 32:

```bash
python -u scripts/run_md22.py \
  --config configs/md22/main.yaml \
  --data-dir data/md22 --outdir outputs/md22_local \
  --methods lite,tera_batched --m 30 \
  --train-batch-size 32 --prediction-batch-size 32 --train-epochs 1 \
  --device cuda --dtype float64 --seeds 7,23,42,71,99
```

Run Standard GP and DDSVGP using their configured training schedules. The
config sets DDSVGP's model dtype to float64 and DSoftKI's to float32.

```bash
python -u scripts/run_md22.py \
  --config configs/md22/main.yaml \
  --data-dir data/md22 --outdir outputs/md22_baselines \
  --methods standard_gp,ddsvgp \
  --device cuda --dtype float64 --seeds 7,23,42,71,99

python -u scripts/run_md22.py \
  --config configs/md22/main.yaml \
  --data-dir data/md22 --outdir outputs/md22_dsoftki \
  --methods dsoftki --device cuda --seeds 7,23,42,71,99
```

### Bayesian optimization

Ackley-500D and Levy-800D use 30 initial observations and a total budget of
100 objective evaluations. The common batch size is 256, with a TERA override
of 32. VBO's settings and TuRBO-LogEI's acquisition settings remain those
in the config.

```bash
python -u scripts/run_bo.py \
  --config configs/bo/main.yaml --outdir outputs/bo_synthetic \
  --benchmarks ackley_500,levy_800 --methods lite,tera_batched,vbo,turbo-logei \
  --seeds 1,27,42,86,99 --budget 100 --n-init 30 --query-batch-size 1 \
  --m 30 --train-batch-size 256 --prediction-batch-size 256 \
  --tera-train-batch-size 32 --tera-prediction-batch-size 32 \
  --device cuda --dtype float64 --verbose
```

The config already sets LITE and TERA acquisition raw samples to 256 and
restarts to 4. To override them together, use `--acq-raw-samples` and
`--acq-restarts`; these flags also set the existing global acquisition options
used by TuRBO-LogEI. VBO uses `--vbo-acq-raw-samples` and `--vbo-acq-restarts`.

For LassoDNA, the DNA dataset is downloaded to `data/lassodna` on the first run.
Add `--lassodna-data-path /path/to/dna.scale` to use an existing copy. `turbo`
uses Thompson sampling; `turbo-logei` above uses LogEI.

```bash
python -u scripts/run_bo.py \
  --config configs/bo/lassodna.yaml --outdir outputs/bo_lassodna \
  --benchmarks lassodna --methods lite,tera_batched,vbo,turbo \
  --seeds 1,27,42,86,99 --budget 100 --n-init 30 --query-batch-size 1 \
  --m 30 --lengthscale-init base --base-lengthscale 1.0 \
  --lengthscale-min 0.003 --lengthscale-max 8.0 \
  --train-batch-size 256 --prediction-batch-size 256 \
  --tera-train-batch-size 32 --tera-prediction-batch-size 32 \
  --initial-train-steps 50 --update-train-steps 10 --refit-every 20 --lr 0.01 \
  --lassodna-backend torch --device cuda --dtype float64 --verbose
```

BO sweeps accept the same run options. The sweep value takes precedence over
the corresponding resolved setting:

```bash
python -u scripts/run_bo_sweep.py \
  --config configs/bo/main.yaml --outdir outputs/bo_m_sweep \
  --methods lite,tera_batched --sweep m --values 10,20,30 \
  --train-batch-size 256 --tera-train-batch-size 32
```

### Configuration overrides

Resolution order is **method CLI > common CLI > method YAML > shared YAML >
defaults**. Flat method keys such as `lite_lr` act as method YAML. Existing flat
global keys retain their original meaning. Common CLI flags update compatible
selected methods; `shared` YAML provides defaults for all compatible methods.
For example, an MD22 config can contain:

```yaml
methods: [lite, tera_batched, vecchia]
shared:
  m: 30
  lr: 0.01
  train_epochs: 1
  train_batch_size: 32
  prediction_batch_size: 32
method_options:
  lite:
    lr: 0.005
  tera:
    prediction_batch_size: 64
  vecchia:
    m: 20
```

`tera` and `tera_batched` share the `tera` option group. BO YAML uses the same
`shared` and `method_options` structure. `--train-steps` selects step-based MD22
training unless epochs are also supplied at the same or a higher priority.
For BO, it sets both initial and update steps unless a stage is overridden at
the same or a higher priority. Boolean flags also accept `--no-…`.

Runs save `config_resolved.yaml`. BO also saves `method_options_resolved.yaml`;
MD22 saves per-dataset, seed and method settings in `method_configs.jsonl`.
