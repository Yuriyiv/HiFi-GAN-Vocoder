import itertools

import torch
import torch.nn as nn


class PITLossWrapper(nn.Module):
    """
    Permutation Invariant Training loss
    """

    def __init__(self, loss_func):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, preds, targets):
        n_sources = preds.shape[1]
        perms = list(itertools.permutations(range(n_sources)))
        losses = torch.stack(
            [self.loss_func(preds[:, perm], targets) for perm in perms], dim=1
        )
        loss = torch.mean(torch.min(losses, dim=1))
        return {"loss": loss}
