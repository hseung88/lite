from contextlib import nullcontext

import torch


def step_scope(name, enabled=False):
    return torch.profiler.record_function(f"step/{name}") if enabled else nullcontext()
