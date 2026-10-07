from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from lite.experiments.md22 import runner
from lite.experiments.md22.observation_noise import config_noise_fractions


def export_parameters(model, cfg, seed, split):
    values = model.lengthscale.detach().cpu().reshape(-1).tolist()
    parameters = {
        "lengthscale": values[0] if len(values) == 1 else values,
        "outputscale": float(model.outputscale),
        "sigma_f": float(model.sigma_f),
        "sigma_g": float(model.sigma_g),
    }
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Fitted lengthscales must be finite and positive.")
    if not math.isfinite(parameters["outputscale"]) or parameters["outputscale"] <= 0:
        raise ValueError("Fitted outputscale must be finite and positive.")
    for name in ("sigma_f", "sigma_g"):
        if not math.isfinite(parameters[name]) or parameters[name] < 0:
            raise ValueError(f"Fitted {name} must be finite and nonnegative.")

    fixed = asdict(cfg)
    fixed.update(parameters)
    fixed.update(
        datasets=[split.name],
        seeds=[seed],
        methods=["vecchia", "tera_batched", "lite"],
        observation_noise_fraction=0.0,
        energy_noise_fraction=0.0,
        force_noise_fraction=0.0,
        initialize_noise_from_observations=False,
        log_training_curves=False,
        tera_gradient_noise_model=model.gradient_noise_model,
        lite_gradient_noise_model=model.gradient_noise_model,
    )
    for prefix in ("vecchia", "tera", "lite"):
        for name in ("train_steps", "train_epochs", "graph_refresh_epochs", "log_every"):
            key = f"{prefix}_{name}"
            if key in fixed:
                fixed[key] = 0
        for name in ("lengthscale", "outputscale", "sigma_f", "sigma_g"):
            key = f"{prefix}_learn_{name}"
            if key in fixed:
                fixed[key] = False

    warm = dict(fixed)
    warm["methods"] = ["lite"]
    for name in ("train_steps", "train_epochs", "graph_refresh_epochs"):
        warm[f"lite_{name}"] = getattr(cfg, f"lite_{name}")
    if warm["lite_train_steps"] <= 0 and warm["lite_train_epochs"] <= 0:
        warm["lite_train_epochs"] = 1
    for name in ("lengthscale", "outputscale", "sigma_f", "sigma_g"):
        warm[f"lite_learn_{name}"] = True

    record = {
        "dataset": split.name,
        "seed": seed,
        "source_method": model.name,
        "source_m": cfg.m,
        "split_id": split.split_id,
        "kernel": cfg.kernel,
        "use_ard": cfg.use_ard,
        "preprocessing_version": cfg.preprocessing_version,
        "x_scale": cfg.x_scale,
        "gradient_noise_model": model.gradient_noise_model,
        "noise_parameter_units": "variance in the MD22 model coordinates",
        "parameters": parameters,
    }
    return record, fixed, warm


def main():
    parser = runner.build_parser()
    parser.description = "Run one clean MD22 fit and export full fitted hyperparameters."
    args = parser.parse_args()
    cfg = runner.config_from_args(args)
    if len(cfg.datasets) != 1 or len(cfg.seeds) != 1 or len(cfg.methods) != 1:
        parser.error("Specify exactly one dataset, one seed, and one method.")
    if cfg.methods[0] not in {"tera_batched", "lite"}:
        parser.error("Use --methods tera_batched for the donor, or lite for a warm-start fit.")
    if config_noise_fractions(cfg) != (0.0, 0.0):
        parser.error("This diagnostic requires clean data: set both noise fractions to zero.")
    if (
        cfg.methods[0] == "tera_batched"
        and cfg.tera_train_steps <= 0
        and cfg.tera_train_epochs <= 0
    ):
        parser.error("The TERA donor must be trained: specify positive training steps or epochs.")

    captured = []

    def capture(model, job_cfg, seed, split):
        captured.append(export_parameters(model, job_cfg, seed, split))

    rows = runner.run(cfg, args.outdir, on_fit=capture)
    if len(rows) != 1 or rows[0].status != "ok" or len(captured) != 1:
        details = "; ".join(row.status for row in rows)
        raise RuntimeError(
            f"The fit/prediction failed; no parameter configurations exported. {details}"
        )

    record, fixed, warm = captured[0]
    out = Path(args.outdir)
    (out / "fitted_hyperparameters.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "fixed_parameters.yaml").write_text(yaml.safe_dump(fixed, sort_keys=False))
    (out / "warmstart_lite.yaml").write_text(yaml.safe_dump(warm, sort_keys=False))
    print(f"Test RMSE per atom: {rows[0].raw_energy_rmse_per_atom:.12g}", flush=True)
    print(f"Exported parameters and fixed/warm-start configurations to {out}", flush=True)


if __name__ == "__main__":
    main()
