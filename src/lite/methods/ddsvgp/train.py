import random
import time

import gpytorch
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lite.methods.common.baseline_util import (
    build_kernel,
    flatten_dataset,
    my_collate_fn,
)
from lite.methods.common.random import RandomStream, rng_scope, seed_for
from lite.methods.ddsvgp.kernel import RBFKernelDirectionalGrad
from lite.methods.ddsvgp.model import DDSVGP


def select_cols_of_y(y_batch, minibatch_dim, dim, *, rng=None):

    idx_y = (random if rng is None else rng).sample(range(1, dim + 1), minibatch_dim)
    idx_y += [0]
    idx_y.sort()
    y_batch = y_batch[:, idx_y]

    derivative_directions = y_batch.new_zeros((minibatch_dim, dim))
    rows = torch.arange(minibatch_dim, device=y_batch.device)
    columns = torch.tensor(idx_y[1:], device=y_batch.device) - 1
    derivative_directions[rows, columns] = 1

    return y_batch, derivative_directions


def train_gp(config, train_dataset, test_dataset, collate_fn=my_collate_fn):
    with rng_scope(int(config.training.seed), device=config.model.device):
        return _train_gp(config, train_dataset, test_dataset, collate_fn)


def _train_gp(config, train_dataset, test_dataset, collate_fn=my_collate_fn):

    dim = train_dataset.dim

    kernel, use_scale, num_inducing, dtype, device, noise, num_directions, mll_type = (
        build_kernel(config.model.kernel),
        config.model.use_scale,
        config.model.num_inducing,
        getattr(torch, config.model.dtype),
        config.model.device,
        config.model.noise,
        config.model.num_directions,
        config.model.mll_type,
    )

    seed, batch_size, num_epochs, lr, lr_sched, gamma = (
        config.training.seed,
        config.training.batch_size,
        config.training.epochs,
        config.training.learning_rate,
        config.training.lr_sched,
        config.training.gamma,
    )

    if config.model.use_ard:
        config.model.kernel.ard_num_dims = train_dataset.dim

        kernel = build_kernel(config.model.kernel)
    assert not config.model.use_ard
    assert isinstance(kernel, RBFKernelDirectionalGrad)
    kernel.lengthscale = config.model.lengthscale

    minibatch_dim = num_directions
    assert num_directions == minibatch_dim

    torch.set_default_dtype(dtype)

    dim = len(train_dataset[0][0])
    n_samples = len(train_dataset)
    num_data = (dim + 1) * n_samples

    if config.model.induce_init == "data":
        inducing_points = torch.zeros(num_inducing, dim)
        for ii in range(num_inducing):
            inducing_points[ii] = train_dataset[ii][0]
        inducing_points = inducing_points.to(device)
    elif config.model.induce_init == "kmeans":
        train_features, train_labels = flatten_dataset(
            train_dataset, batch_size=256, collate_fn=collate_fn
        )
        kmeans = KMeans(
            n_clusters=min(len(train_features), num_inducing),
            random_state=seed_for(seed, RandomStream.INITIALIZATION),
        )
        kmeans.fit(train_features)
        centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(centers).to(dtype=dtype, device=device)
    else:
        inducing_points = torch.rand(num_inducing, dim)
        inducing_points = inducing_points.to(device)
    inducing_directions = torch.eye(dim, device=device, dtype=dtype)[:num_directions]
    inducing_directions = inducing_directions.repeat(num_inducing, 1)

    inducing_points = inducing_points.to(device=device)
    inducing_directions = inducing_directions.to(device=device)

    model = DDSVGP(
        inducing_points,
        inducing_directions,
        kernel,
        use_scale=use_scale,
        learn_inducing_locations=True,
    ).to(device=device, dtype=dtype)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device, dtype=dtype)
    likelihood.noise = torch.tensor([noise]).to(device=device)

    model.train()
    likelihood.train()

    variational_optimizer = torch.optim.Adam(
        [
            {"params": model.variational_parameters()},
        ],
        lr=lr,
    )
    hyperparameter_optimizer = torch.optim.Adam(
        [
            {"params": model.hyperparameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    if lr_sched == "step_lr":
        num_batches = int(np.ceil(n_samples / batch_size))
        milestones = [int(num_epochs * num_batches / 3), int(2 * num_epochs * num_batches / 3)]
        hyperparameter_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            hyperparameter_optimizer, milestones, gamma=gamma
        )
        variational_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            variational_optimizer, milestones, gamma=gamma
        )
    elif lr_sched is None:

        def lr_sched(epoch):
            return 1.0

        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(
            hyperparameter_optimizer, lr_lambda=lr_sched
        )
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(
            variational_optimizer, lr_lambda=lr_sched
        )
    else:
        hyperparameter_scheduler = torch.optim.lr_scheduler.LambdaLR(
            hyperparameter_optimizer, lr_lambda=lr_sched
        )
        variational_scheduler = torch.optim.lr_scheduler.LambdaLR(
            variational_optimizer, lr_lambda=lr_sched
        )

    if mll_type == "ELBO":
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    elif mll_type == "PLL":
        mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=num_data)

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
    direction_rng = random.Random(seed_for(seed, RandomStream.DERIVATIVE_SUBSET))
    for epoch in tqdm(range(num_epochs)):
        t1 = time.perf_counter()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_batch, derivative_directions = select_cols_of_y(
                y_batch, minibatch_dim, dim, rng=direction_rng
            )

            kwargs = {}

            kwargs["derivative_directions"] = derivative_directions.repeat(y_batch.size(0), 1)

            y_batch = y_batch.reshape(torch.numel(y_batch))

            variational_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            # The MLL applies the likelihood itself; pass q(f), not likelihood(q(f)).
            output = model(x_batch, **kwargs)
            loss = -mll(output, y_batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("DDSVGP training loss is not finite.")
            loss.backward()

            variational_optimizer.step()
            variational_scheduler.step()
            hyperparameter_optimizer.step()
            hyperparameter_scheduler.step()
            global_step += 1

            if (
                test_dataset is not None
                and curve_log_every > 0
                and global_step % curve_log_every == 0
            ):
                eval_results = eval_gp(
                    model,
                    likelihood,
                    test_dataset,
                    num_directions,
                    device=device,
                    num_workers=config.dataset.num_workers,
                    collate_fn=collate_fn,
                )
                training_history.append(
                    {
                        "step": float(global_step),
                        "normalized_energy_rmse_per_atom": float(eval_results["rmse"]),
                        "grad_rmse": float(eval_results["d_rmse"]),
                    }
                )

        t2 = time.perf_counter()
        if test_dataset is not None and curve_log_every <= 0:
            eval_results = eval_gp(
                model,
                likelihood,
                test_dataset,
                num_directions,
                device=device,
                num_workers=config.dataset.num_workers,
                collate_fn=collate_fn,
            )
            training_history.append(
                {
                    "step": float(epoch + 1),
                    "normalized_energy_rmse_per_atom": float(eval_results["rmse"]),
                    "grad_rmse": float(eval_results["d_rmse"]),
                    "epoch_time": float(t2 - t1),
                }
            )

    model.training_history = training_history

    return model, likelihood


def eval_gp(
    model,
    likelihood,
    test_dataset,
    num_directions,
    batch_size=256,
    device="cuda:0",
    num_workers=8,
    collate_fn=None,
) -> float:
    dim = test_dataset.dim
    squared_diffs = []
    squared_d_diffs = []
    nlls = []

    kwargs = {}
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        derivative_directions = torch.eye(dim, device=x_batch.device, dtype=x_batch.dtype)[
            :num_directions
        ]
        derivative_directions = derivative_directions.repeat(len(x_batch), 1)
        kwargs["derivative_directions"] = derivative_directions

        x_batch = x_batch.requires_grad_()
        preds = likelihood(model(x_batch, **kwargs))
        means = preds.mean[:: num_directions + 1]
        stds = preds.variance.sqrt()[:: num_directions + 1]
        nll = -torch.distributions.Normal(means, stds).log_prob(y_batch[:, 0])
        grad = torch.autograd.grad(
            outputs=means, inputs=x_batch, grad_outputs=torch.ones_like(means)
        )[0]

        squared_diffs += [(means - y_batch[:, 0]).detach().cpu() ** 2]
        squared_d_diffs += [(grad.reshape(-1) - y_batch[:, 1:].reshape(-1)).detach().cpu() ** 2]
        nlls += [nll.detach().cpu()]
    rmse = torch.sqrt(torch.sum(torch.cat(squared_diffs)) / len(test_dataset)).item()
    d_rmse = torch.sqrt(torch.sum(torch.cat(squared_d_diffs)) / len(test_dataset)).item()
    nll = torch.cat(nlls).mean()
    print(
        "RMSE:",
        rmse,
        "D_RMSE",
        d_rmse,
        "NLL",
        nll,
        "NOISE",
        likelihood.noise_covar.noise.cpu(),
        "LENGTHSCALE",
        model.get_lengthscale(),
        "OUTPUTSCALE",
        model.get_outputscale(),
    )

    return {
        "rmse": rmse,
        "d_rmse": d_rmse,
        "nll": nll,
    }
