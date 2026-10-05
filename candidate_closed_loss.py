from typing import Sequence

import torch
import torch.nn.functional as F


def slate_listwise_loss(scores: torch.Tensor, gains: torch.Tensor, tau: float) -> torch.Tensor:
    gains = gains.to(device=scores.device, dtype=scores.dtype)
    if torch.sum(gains) <= 0:
        return scores.sum() * 0.0
    else:
        target = gains / torch.sum(gains)
    log_probs = F.log_softmax(scores / max(float(tau), 1e-6), dim=0)
    return -(target * log_probs).sum()


def slate_pairwise_odds_loss(scores: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    gains = gains.to(device=scores.device, dtype=scores.dtype)
    losses = []
    for left in range(scores.numel()):
        for right in range(scores.numel()):
            gain_delta = gains[left] - gains[right]
            if gain_delta <= 0:
                continue
            losses.append(gain_delta * F.softplus(scores[right] - scores[left]))
    if not losses:
        return scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def candidate_closed_loss(
    grouped_scores: Sequence[torch.Tensor],
    batch_slates: Sequence[dict],
    tau: float,
    odds_weight: float,
) -> torch.Tensor:
    losses = []
    for scores, slate in zip(grouped_scores, batch_slates):
        gains = torch.tensor(slate["gains"], dtype=torch.float32, device=scores.device)
        listwise = slate_listwise_loss(scores, gains, tau=tau)
        pairwise = slate_pairwise_odds_loss(scores, gains)
        losses.append(listwise + float(odds_weight) * pairwise)
    if not losses:
        raise ValueError("candidate_closed_loss received an empty batch")
    return torch.stack(losses).mean()
