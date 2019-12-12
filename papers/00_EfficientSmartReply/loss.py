"""Mean Negative Loss."""

import torch
import torch.nn as nn


class ApproxMeanNegativeLoss(nn.Module):
    """Loss function."""

    def __init__(self):
        super(ApproxMeanNegativeLoss, self).__init__()

    def forward(self, src_pos, trg_pos, batch_size):
        try:
            assert batch_size == src_pos.size()[0]
        except AssertionError:
            batch_size = src_pos.size()[0]
        S_xi_yi = torch.mm(src_pos, trg_pos.t()).diag()
        log_sum_exp_S = torch.log(
            torch.sum(torch.exp(torch.mm(src_pos, trg_pos.t())), dim=1))
        return -(((S_xi_yi - log_sum_exp_S).sum()) / batch_size) + 1e-9
