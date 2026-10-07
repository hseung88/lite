from __future__ import annotations

from omegaconf import OmegaConf

from lite.methods.names import normalize_md22_method_name


def _dsoftki_kernel_target(kernel: str) -> str:
    if kernel == "rbf":
        return "RBFKernel"
    if kernel == "matern52":
        return "MaternKernel"
    raise ValueError(f"Unsupported DSoftKI kernel {kernel!r}.")


def build_config(args, meta):
    method = normalize_md22_method_name(args.method)
    if method == "dsoftki":
        cfg = {
            "dataset": {"name": meta.dataset, "num_workers": args.num_workers},
            "model": {
                "name": "dsoftki",
                "kernel": {
                    "_target_": _dsoftki_kernel_target(args.kernel),
                    "lengthscale": args.lengthscale,
                    "nu": 2.5,
                    "ard_num_dims": None,
                },
                "embed_dim": -1,
                "hidden_dim": 64,
                "per_interp_T": True,
                "min_T": 0.00005,
                "use_dot": True,
                "grad_only": False,
                "lengthscale": args.lengthscale,
                "use_ard": args.use_ard,
                "use_scale": True,
                "num_interp": args.num_inducing,
                "interp_init": "kmeans",
                "noise": args.noise,
                "deriv_noise": args.deriv_noise,
                "learn_noise": args.learn_noise,
                "solver": args.solver,
                "cg_tolerance": args.cg_tolerance,
                "mll_approx": args.mll_approx,
                "fit_chunk_size": args.fit_chunk_size,
                "use_qr": args.use_qr,
                "dtype": args.dtype,
                "device": args.device,
                "fit_device": args.fit_device,
                "skip_nll": True,
            },
            "training": {
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "embed_lr": args.embed_lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "curve_log_every": args.curve_log_every if args.log_training_curves else 0,
            },
        }
    elif method == "ddsvgp":
        cfg = {
            "dataset": {"name": meta.dataset, "num_workers": args.num_workers},
            "model": {
                "name": "ddsvgp",
                "kernel": {"_target_": "RBFKernelDirectionalGrad", "ard_num_dims": None},
                "lengthscale": args.lengthscale,
                "use_scale": True,
                "use_ard": False,
                "num_inducing": args.num_inducing,
                "num_directions": args.num_directions,
                "induce_init": "kmeans",
                "noise": args.noise,
                "dtype": args.dtype,
                "device": args.device,
                "mll_type": args.mll_type,
            },
            "training": {
                "seed": args.seed,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "gamma": args.gamma,
                "lr_sched": None,
                "curve_log_every": args.curve_log_every if args.log_training_curves else 0,
            },
        }
    else:
        raise ValueError(f"Unsupported method {args.method}")
    return OmegaConf.create(cfg)
