import time

import torch
from omegaconf.dictconfig import DictConfig
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lite.methods.common.baseline_util import (
    build_kernel,
    filter_param,
    flatten_dataset,
    my_collate_fn,
)
from lite.methods.common.random import RandomStream, rng_scope, seed_for
from lite.methods.dsoftki.model import DSoftKI


def train_gp(
    config: DictConfig,
    train_dataset: Dataset,
    test_dataset: Dataset | None,
    collate_fn=my_collate_fn,
) -> DSoftKI:
    with rng_scope(int(config.training.seed), device=config.model.device):
        return _train_gp(config, train_dataset, test_dataset, collate_fn)


def _train_gp(
    config: DictConfig,
    train_dataset: Dataset,
    test_dataset: Dataset | None,
    collate_fn=my_collate_fn,
) -> DSoftKI:

    (
        kernel,
        use_scale,
        num_interp,
        interp_init,
        dtype,
        device,
        fit_device,
        noise,
        learn_noise,
        deriv_noise,
        solver,
        cg_tolerance,
        mll_approx,
        fit_chunk_size,
        use_qr,
    ) = (
        build_kernel(config.model.kernel),
        config.model.use_scale,
        config.model.num_interp,
        config.model.interp_init,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.fit_device,
        float(config.model.noise),
        config.model.learn_noise,
        float(config.model.deriv_noise),
        config.model.solver,
        float(config.model.cg_tolerance),
        config.model.mll_approx,
        config.model.fit_chunk_size,
        config.model.use_qr,
    )

    per_interp_T, min_T, embed_dim, hidden_dim, use_dot = (
        config.model.per_interp_T,
        config.model.min_T,
        config.model.embed_dim,
        config.model.hidden_dim,
        config.model.use_dot,
    )
    embed_dim = min(embed_dim, round(train_dataset.dim / 2))

    if config.model.use_ard:
        if embed_dim == -1:
            config.model.kernel.ard_num_dims = train_dataset.dim
        else:
            config.model.kernel.ard_num_dims = embed_dim
        kernel = build_kernel(config.model.kernel)

    seed, batch_size, epochs, lr = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
    )

    train_features, train_labels = flatten_dataset(
        train_dataset, batch_size=config.model.fit_chunk_size, collate_fn=collate_fn
    )

    if interp_init == "kmeans":
        if embed_dim != -1:
            print(f"Warning: k-means initialization not supported with embed_dim={embed_dim}")
            print(f"Falling back to random initialization in embedding space (dim={embed_dim})")
            interp_points = torch.rand(num_interp, embed_dim, dtype=dtype, device=device)
        else:
            kmeans = KMeans(
                n_clusters=min(len(train_features), num_interp),
                random_state=seed_for(seed, RandomStream.INITIALIZATION),
            )
            kmeans.fit(train_features)
            centers = kmeans.cluster_centers_
            interp_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        if embed_dim != -1:
            interp_points = torch.rand(num_interp, embed_dim, dtype=dtype, device=device)

        else:
            interp_points = torch.rand(num_interp, train_dataset.dim, dtype=dtype, device=device)

    if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
        kernel.lengthscale = config.model.lengthscale

    model = DSoftKI(
        kernel,
        interp_points,
        train_dataset.dim,
        dtype=dtype,
        device=device,
        fit_device=fit_device,
        noise=noise,
        learn_noise=learn_noise,
        deriv_noise=deriv_noise,
        use_scale=use_scale,
        solver=solver,
        cg_tolerance=cg_tolerance,
        mll_approx=mll_approx,
        fit_chunk_size=fit_chunk_size,
        use_qr=use_qr,
        grad_only=config.model.grad_only,
        per_interp_T=per_interp_T,
        min_T=min_T,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        use_dot=use_dot,
    )

    if embed_dim != -1:
        embed_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "embedding" in name:
                embed_params.append(param)
            else:
                if learn_noise and name != "likelihood.noise_covar.raw_noise":
                    other_params.append(param)
        optimizer = torch.optim.Adam(
            [
                {"params": embed_params, "lr": config.training.embed_lr},
                {"params": other_params, "lr": lr},
            ]
        )
    else:
        if learn_noise:
            params = model.parameters()
        else:
            params = filter_param(model.named_parameters(), "likelihood.noise_covar.raw_noise")
        optimizer = torch.optim.Adam(params, lr=lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed_for(seed, RandomStream.TRAINING_BATCH)),
        num_workers=config.dataset.num_workers,
        collate_fn=collate_fn,
    )
    training_history = []
    curve_log_every = int(getattr(config.training, "curve_log_every", 0) or 0)
    global_step = 0
    pbar = tqdm(range(epochs), desc="Optimizing MLL")
    for epoch in pbar:
        t1 = time.perf_counter()

        neg_mlls = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.clone().detach().to(dtype=dtype, device=device)

            if config.model.grad_only:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)[:, 1:]
            else:
                y_batch = y_batch.clone().detach().to(dtype=dtype, device=device)

            optimizer.zero_grad()
            neg_mll = -model.mll(x_batch, y_batch)
            if not bool(torch.isfinite(neg_mll)):
                raise FloatingPointError("DSoftKI training loss is not finite.")
            neg_mlls += [-neg_mll.item()]
            neg_mll.backward()

            if embed_dim != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            update = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        update = False
                        break
            if update:
                optimizer.step()
            global_step += 1

            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
            pbar.set_postfix(MLL=f"{-neg_mll.item()}")
            if (
                test_dataset is not None
                and curve_log_every > 0
                and global_step % curve_log_every == 0
            ):
                t_fit0 = time.perf_counter()
                model.fit(train_features, train_labels)
                t_fit1 = time.perf_counter()
                results = eval_gp(
                    model,
                    test_dataset,
                    device=device,
                    num_workers=config.dataset.num_workers,
                    collate_fn=collate_fn,
                    grad_only=config.model.grad_only,
                    skip_nll=config.model.skip_nll,
                )
                training_history.append(
                    {
                        "step": float(global_step),
                        "normalized_energy_rmse_per_atom": float(results["rmse"]),
                        "grad_rmse": float(results["d_rmse"]),
                        "fit_time": float(t_fit1 - t_fit0),
                    }
                )
        t2 = time.perf_counter()

        model.fit(train_features, train_labels)
        t3 = time.perf_counter()

        if test_dataset is not None and curve_log_every <= 0:
            results = eval_gp(
                model,
                test_dataset,
                device=device,
                num_workers=config.dataset.num_workers,
                collate_fn=collate_fn,
                grad_only=config.model.grad_only,
                skip_nll=config.model.skip_nll,
            )
            training_history.append(
                {
                    "step": float(epoch + 1),
                    "normalized_energy_rmse_per_atom": float(results["rmse"]),
                    "grad_rmse": float(results["d_rmse"]),
                    "epoch_time": float(t2 - t1),
                    "fit_time": float(t3 - t2),
                }
            )

    model.training_history = training_history

    return model


def eval_gp(
    model: DSoftKI,
    test_dataset: Dataset,
    device="cuda:0",
    num_workers=8,
    collate_fn=None,
    grad_only=False,
    skip_nll=False,
    skip_nll_full=False,
) -> float:
    with torch.no_grad():
        squared_errors = []
        squared_d_errors = []
        nlls = []
        nlls_val = []
        nlls_grad = []
        if test_dataset.dim <= 3:
            batch_size = 4096
        elif test_dataset.dim <= 10:
            batch_size = 2048
        elif test_dataset.dim <= 20:
            batch_size = 512
        else:
            batch_size = 128
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        for idx, (x_batch, y_batch) in tqdm(enumerate(test_loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = len(x_batch)
            if grad_only:
                d_ys = y_batch[:, 1:]
                y = d_ys.reshape(-1)
                y_preds = model.pred(x_batch)
                squared_d_errors += [(y_preds - y).detach().cpu() ** 2]
            else:
                ys = y_batch[:, 0]
                d_ys = y_batch[:, 1:]
                y_preds = model.pred(x_batch)
                y_pred = y_preds[0:B]
                squared_errors += [(y_pred - ys).detach().cpu() ** 2]
                squared_d_errors += [(y_preds[B:] - d_ys.reshape(-1)).detach().cpu() ** 2]

            if skip_nll:
                nlls += [torch.zeros(1)]
                nlls_val += [torch.zeros(1)]
                nlls_grad += [torch.zeros(1)]
            else:
                std = model.pred_cov_val(x_batch).diag().sqrt()
                try:
                    nlls += [-torch.distributions.Normal(y_pred, std).log_prob(ys).detach().cpu()]
                except ValueError:
                    nlls += [torch.nan]
                del std
                if skip_nll_full:
                    nlls_val += [torch.zeros(1)]
                    nlls_grad += [torch.zeros(1)]
                else:
                    cov = model.pred_cov(x_batch)
                    std2 = cov[0:B].diag().sqrt()
                    std3 = cov[B:].diag().sqrt()
                    nlls_val += [
                        -torch.distributions.Normal(y_pred, std2).log_prob(ys).detach().cpu()
                    ]
                    nlls_grad += [
                        -torch.distributions.Normal(y_preds[B:], std3)
                        .log_prob(d_ys.reshape(-1))
                        .detach()
                        .cpu()
                    ]
                    del cov, std2, std3

            del x_batch, y_batch
            torch.cuda.empty_cache()

        if grad_only:
            rmse = 0
            d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_errors)) / len(test_dataset)).item()
            nll = torch.cat(nlls).mean()
            nll_val = torch.cat(nlls_val).mean()
            nll_grad = torch.cat(nlls_grad).mean()

            print(
                "D_RMSE",
                d_rmse,
                "NLL",
                nll.item(),
                "NLL_val",
                nll_val.item(),
                "NLL_grad",
                nll_grad.item(),
                "NOISE",
                model.noise.cpu().item(),
                "DNOISE",
                model.deriv_noise.cpu().item(),
                "LENGTHSCALE",
                model.get_lengthscale(),
                "OUTPUTSCALE",
                model.get_outputscale(),
                "T",
                model.T.cpu(),
            )
        else:
            rmse = torch.sqrt(torch.sum(torch.cat(squared_errors)) / len(test_dataset)).item()
            d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_errors)) / len(test_dataset)).item()
            nll = torch.cat(nlls).mean()
            nll_val = torch.cat(nlls_val).mean()
            nll_grad = torch.cat(nlls_grad).mean()

            print(
                "RMSE:",
                rmse,
                "D_RMSE",
                d_rmse,
                "NLL",
                nll.item(),
                "NLL_val",
                nll_val.item(),
                "NLL_grad",
                nll_grad.item(),
                "NOISE",
                model.noise.cpu().item(),
                "DNOISE",
                model.deriv_noise.cpu().item(),
                "LENGTHSCALE",
                model.get_lengthscale(),
                "OUTPUTSCALE",
                model.get_outputscale(),
                "T",
                model.T.cpu(),
            )

    return {
        "rmse": rmse,
        "d_rmse": d_rmse,
        "nll": nll,
        "nll_val": nll_val,
        "nll_grad": nll_grad,
    }
