import torch

from h2df.losses import (
    ga_loss,
    h2df_loss,
    masked_representation_mse,
    npo_loss,
    per_example_causal_loss,
    simnpo_loss,
    smooth_hinge,
)


def test_smooth_hinge_approximates_positive_part():
    values = torch.tensor([-2.0, 0.0, 2.0])
    result = smooth_hinge(values, tau=0.01)
    assert result[0] < 1e-4
    assert result[1] > 0
    assert torch.isclose(result[2], torch.tensor(2.0), atol=1e-4)


def test_h2df_one_sided_pressure_decreases_after_target():
    target = torch.tensor([2.0])
    below = h2df_loss(torch.tensor([0.0]), target, tau=0.25)
    above = h2df_loss(torch.tensor([4.0]), target, tau=0.25)
    assert below > above


def test_h2df_two_sided_penalizes_both_sides():
    lower = torch.tensor([1.0])
    upper = torch.tensor([3.0])
    inside = h2df_loss(torch.tensor([2.0]), lower, 0.1, upper, gamma=1.0)
    below = h2df_loss(torch.tensor([0.0]), lower, 0.1, upper, gamma=1.0)
    above = h2df_loss(torch.tensor([4.0]), lower, 0.1, upper, gamma=1.0)
    assert below > inside
    assert above > inside


def test_ga_rewards_higher_forget_loss():
    assert ga_loss(torch.tensor([2.0])) < ga_loss(torch.tensor([1.0]))


def test_npo_rewards_lower_log_probability_than_reference():
    original = torch.tensor([1.0])
    not_forgotten = npo_loss(torch.tensor([1.0]), original, beta=0.1)
    forgotten = npo_loss(torch.tensor([3.0]), original, beta=0.1)
    assert forgotten < not_forgotten


def test_simnpo_saturates_as_nll_exceeds_margin():
    not_forgotten = simnpo_loss(torch.tensor([1.0]), beta=2.5, gamma=2.0)
    forgotten = simnpo_loss(torch.tensor([4.0]), beta=2.5, gamma=2.0)
    assert forgotten < not_forgotten


def test_masked_representation_mse_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [100.0, 100.0]]])
    target = torch.zeros_like(hidden)
    mask = torch.tensor([[1, 0]])
    assert torch.isclose(masked_representation_mse(hidden, target, mask), torch.tensor(1.0))


def test_per_example_loss_ignores_prompt_and_padding():
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, -100]])
    loss = per_example_causal_loss(logits, labels)
    assert torch.isclose(loss, torch.tensor([torch.log(torch.tensor(3.0))])).all()
