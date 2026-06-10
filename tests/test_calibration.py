from h2df.calibration import assign_bin, build_targets, quantile_edges


def test_quantile_bins_are_monotonic():
    edges = quantile_edges([1, 2, 3, 4, 5, 6, 7, 8], bins=4)
    assert edges == sorted(edges)
    assert assign_bin(1, edges) == 0
    assert assign_bin(8, edges) == len(edges)


def test_tofu_target_respects_minimum_displacement():
    calibration = [
        {
            "token_length": 10,
            "original_loss": value,
            "domain": "default",
            "question_type": "what",
        }
        for value in [0.5, 1.0, 1.5, 2.0]
    ]
    forget = [
        {
            "token_length": 10,
            "original_loss": 3.0,
            "domain": "default",
            "question_type": "what",
        }
    ]
    targets, _ = build_targets(
        forget,
        calibration,
        "tofu",
        {"bins": 1, "min_bin_size": 1, "rho": 0.5, "delta_min": 0.5},
    )
    assert targets[0]["lower_target"] == 3.5


def test_muse_builds_ordered_band():
    calibration = [
        {"token_length": 10, "original_loss": value, "domain": "books"}
        for value in range(1, 11)
    ]
    forget = [{"token_length": 10, "original_loss": 1.0, "domain": "books"}]
    targets, _ = build_targets(
        forget,
        calibration,
        "muse",
        {
            "bins": 1,
            "min_bin_size": 1,
            "rho_low": 0.5,
            "rho_high": 0.9,
        },
    )
    assert targets[0]["lower_target"] < targets[0]["upper_target"]
