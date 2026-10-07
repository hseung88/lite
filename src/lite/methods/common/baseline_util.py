from typing import Any, Tuple

import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def my_collate_fn(batch):
    data = [item[0] for item in batch]
    efs = [torch.cat([torch.tensor([item[1]["energy"]]), item[1]["neg_force"]]) for item in batch]
    data_tensor = torch.stack(data, dim=0)
    efs_tensor = torch.stack(efs, dim=0)
    return data_tensor, efs_tensor


def flatten_dataset(
    dataset: Dataset, collate_fn=None, batch_size=8192
) -> Tuple[torch.Tensor, torch.Tensor]:
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    train_x = []
    train_y = []
    for batch_x, batch_y in tqdm(train_loader):
        train_x += [batch_x]
        train_y += [batch_y]
    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0).squeeze(-1)
    return train_x, train_y


def filter_param(
    named_params: list[Tuple[str, torch.nn.Parameter]], name: str
) -> list[Tuple[str, torch.nn.Parameter]]:
    return [param for n, param in named_params if n != name]


def build_kernel(config: DictConfig | dict) -> Any:
    from gpytorch.kernels import LinearKernel, MaternKernel, RBFKernel, RBFKernelGrad, ScaleKernel

    from lite.methods.ddsvgp.kernel import RBFKernelDirectionalGrad

    kernels = {
        cls.__name__: cls
        for cls in (
            LinearKernel,
            MaternKernel,
            RBFKernel,
            RBFKernelGrad,
            ScaleKernel,
            RBFKernelDirectionalGrad,
        )
    }
    name = config["_target_"]
    if name not in kernels:
        raise ValueError(f"Unknown kernel: {name!r}. Expected one of {tuple(kernels)}")
    return kernels[name](**{k: v for k, v in config.items() if k != "_target_"})
