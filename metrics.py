from typing import List
import numpy as np
import torch
from pyro.ops.stats import crps_empirical


def eval_rrmse(preds, labels):
    # MSE
    loss = (preds-labels)**2
    # Nomalizer
    loss_norm = (labels.mean()-labels)**2
    return torch.sqrt(torch.sum(loss))/torch.sqrt(torch.sum(loss_norm))


def eval_crps_mean(preds, labels):
    """
    preds: (n_sample, N, Q)
    labels: (N, Q)
    """
    crps = crps_empirical(preds, labels)  # (N, Q)
    crps /= labels.abs().mean()
    return crps.mean().item(), crps


def eval_crps(preds, labels):
    """
    preds: (n_sample, N, Q)
    labels: (N, Q)
    """
    crps = crps_empirical(preds, labels)
    crps /= labels.abs().mean(-1, keepdim=True)
    return crps.mean().item(), crps


def eval_crps_sum(preds, labels):
    """
    preds: (n_sample, N, Q)
    labels: (N, Q)
    """
    preds_sum = preds.sum(-2)
    labels_sum = labels.sum(-2)
    crps_sum = crps_empirical(preds_sum, labels_sum)  # (Q,)
    crps_sum /= labels_sum.abs().mean()
    return crps_sum.mean().item(), crps_sum


def eval_energy_score(preds, labels, beta=1.0):
    """
    preds: (n_sample, N, Q)
    labels: (N, Q)
    """
    assert 0 < beta < 2
    num_samples = preds.shape[0]
    # The Frobenius norm of a matrix is equal to the Euclidean norm of its element:
    # the square root of the sum of the square of its elements
    norm = torch.linalg.norm(preds - labels[None, :, :], ord="fro", dim=(1, 2))
    first_term = (norm**beta).mean()

    # For the second term of the energy score, we need two independant realizations of the distributions.
    # So we do a sum ignoring the i == j terms (which would be zero anyway), and we normalize by the
    # number of pairs of samples we have summed over.
    s = torch.tensor(0.0)
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            norm = torch.linalg.norm(preds[i] - preds[j], ord="fro")
            s += norm**beta
    second_term = s / (num_samples * (num_samples - 1) / 2)

    return first_term - 0.5 * second_term


class QuantileLoss():
    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor):
        quantiles = torch.quantile(y_pred, torch.tensor(self.quantiles, device=y_pred.device), dim=-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - quantiles[i]
            losses.append(torch.max((q-1)*errors, q*errors))
        return losses
    

def get_metrics(preds, labels):
    """
    preds: (n_frcs, N, Q, n_sample)
    labels: (n_frcs, N, Q)
    """
    ql = QuantileLoss(quantiles=[0.5, 0.9])
    n_frcs = labels.shape[0]
    acc_metrics = []
    crps_mean_noagg_all, crps_noagg_all, crps_sum_noagg_all = [], [], []
    for i in range(n_frcs):
        crps_mean, crps_mean_noagg = eval_crps_mean(preds[i].permute(2, 0, 1), labels[i])
        crps, crps_noagg = eval_crps(preds[i].permute(2, 0, 1), labels[i])
        crps_sum, crps_sum_noagg = eval_crps_sum(preds[i].permute(2, 0, 1), labels[i])

        crps_mean_noagg_all.append(crps_mean_noagg)
        crps_noagg_all.append(crps_noagg)
        crps_sum_noagg_all.append(crps_sum_noagg)
        
        es = eval_energy_score(preds[i].permute(2, 0, 1), labels[i]).item()

        ql05, ql09 = ql.loss(preds[i], labels[i])
        p05_risk = (ql05.sum()/labels[i].sum()).item()
        p09_risk = (ql09.sum()/labels[i].sum()).item()

        rrmse = eval_rrmse(preds[i].mean(-1), labels[i])

        acc_metrics.append([crps_mean, crps, crps_sum, p05_risk, p09_risk, es, rrmse])

    crps_mean_noagg = torch.stack(crps_mean_noagg_all)
    crps_noagg = torch.stack(crps_noagg_all)
    crps_sum_noagg = torch.stack(crps_sum_noagg_all)

    return np.array(acc_metrics).sum(0) / n_frcs, crps_mean_noagg, crps_noagg, crps_sum_noagg