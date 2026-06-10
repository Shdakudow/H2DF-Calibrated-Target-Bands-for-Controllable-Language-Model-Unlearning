from __future__ import annotations

import torch
import torch.nn.functional as F


def per_example_causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean teacher-forced cross-entropy per sequence, ignoring label -100."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction="none",
        ignore_index=-100,
    )
    valid = shift_labels.ne(-100)
    counts = valid.sum(dim=1).clamp_min(1)
    return (token_losses * valid).sum(dim=1) / counts


def smooth_hinge(value: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return tau * F.softplus(value / tau)


def h2df_loss(
    scores: torch.Tensor,
    lower_targets: torch.Tensor,
    tau: float,
    upper_targets: torch.Tensor | None = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    losses = smooth_hinge(lower_targets - scores, tau)
    if upper_targets is not None and gamma > 0:
        losses = losses + gamma * smooth_hinge(scores - upper_targets, tau)
    return losses.mean()


def ga_loss(scores: torch.Tensor) -> torch.Tensor:
    return -scores.mean()


def npo_loss(scores: torch.Tensor, original_scores: torch.Tensor, beta: float) -> torch.Tensor:
    """NPO using length-normalized sequence log probabilities.

    log p_theta - log p_ref = -score_theta + score_ref.
    """
    if beta <= 0:
        raise ValueError("NPO beta must be positive")
    log_ratio = -scores + original_scores
    return (2.0 / beta) * F.softplus(beta * log_ratio).mean()


def simnpo_loss(scores: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    """Official SimNPO form using length-normalized NLL scores.

    Equivalent to -2/beta * log(sigmoid(beta * (NLL - gamma))).
    """
    if beta <= 0:
        raise ValueError("SimNPO beta must be positive")
    return (2.0 / beta) * F.softplus(-beta * (scores - gamma)).mean()


def masked_representation_mse(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared representation error over non-padding token positions."""
    token_error = (hidden_states.float() - targets.float()).pow(2).mean(dim=-1)
    mask = attention_mask.to(token_error.dtype)
    return (token_error * mask).sum() / mask.sum().clamp_min(1.0)


def lora_l2(model: torch.nn.Module) -> torch.Tensor:
    terms = [
        parameter.float().pow(2).sum()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if not terms:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(terms).sum()
