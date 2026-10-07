from __future__ import annotations

import torch


@torch.no_grad()
def maximin_ordering(x_scaled: torch.Tensor) -> torch.Tensor:
    n = len(x_scaled)
    order = torch.empty(n, dtype=torch.long, device=x_scaled.device)
    if n == 0:
        return order
    norms = x_scaled.square().sum(-1)
    minimum = norms.new_full((n,), float("inf"))
    selected = torch.zeros(n, dtype=torch.bool, device=x_scaled.device)
    index = torch.zeros(1, dtype=torch.long, device=x_scaled.device)
    for step in range(n):
        order[step : step + 1] = index
        point = x_scaled.index_select(0, index).squeeze(0)
        distances = (norms + norms.index_select(0, index) - 2 * (x_scaled @ point)).clamp_min_(0)
        torch.minimum(minimum, distances, out=minimum)
        selected.index_fill_(0, index, True)
        minimum.masked_fill_(selected, -1)
        index = minimum.argmax().reshape(1)
    return order


@torch.no_grad()
def predecessor_neighbors(x_scaled: torch.Tensor, m: int, block_size: int = 256) -> torch.Tensor:
    n = len(x_scaled)
    k = min(max(int(m), 0), max(n - 1, 0))
    neighbors = torch.zeros((n, k), dtype=torch.long, device=x_scaled.device)
    if k == 0:
        return neighbors
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        distances = torch.cdist(
            x_scaled[start:stop], x_scaled[:stop], compute_mode="donot_use_mm_for_euclid_dist"
        )
        rows = torch.arange(start, stop, device=x_scaled.device)[:, None]
        columns = torch.arange(stop, device=x_scaled.device)[None, :]
        distances.masked_fill_(columns >= rows, float("inf"))
        count = min(k, stop)
        indices = distances.topk(count, largest=False).indices
        neighbors[start:stop, :count] = indices
    return neighbors


def knn_to_eval(
    x_train_scaled: torch.Tensor, x_eval_scaled: torch.Tensor, m: int
) -> list[torch.Tensor]:
    if m <= 0:
        return [
            torch.empty(0, dtype=torch.long, device=x_train_scaled.device)
            for _ in range(x_eval_scaled.shape[0])
        ]
    dists = torch.cdist(x_eval_scaled, x_train_scaled)
    k = min(m, x_train_scaled.shape[0])
    nn = torch.topk(dists, k=k, largest=False).indices
    return [nn[i].contiguous() for i in range(nn.shape[0])]
