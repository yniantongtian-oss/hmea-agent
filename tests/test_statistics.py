import json

import numpy as np
import pytest

from hmea.statistics import paired_mean_difference


def test_constant_offset_preserves_pairing_and_reports_zero_variation():
    reference = np.array([100.0, 200.0, 300.0, 400.0])
    candidate = reference + 1.0

    result = paired_mean_difference(candidate, reference, n_resamples=251, seed=8)

    assert result["n_pairs"] == 4
    assert result["mean_difference"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["ci95_high"] == 1.0
    assert result["observed_zero_variance"] is True
    assert result["per_seed_differences"] == [1.0, 1.0, 1.0, 1.0]


def test_identical_pairs_have_exact_zero_difference():
    values = np.array([0.25, 2.0, 7.0])

    result = paired_mean_difference(values, values)

    assert result["mean_difference"] == 0.0
    assert result["ci95_low"] == result["ci95_high"] == 0.0
    assert result["observed_zero_variance"] is True


def test_variable_pairs_report_native_unit_mean_and_raw_differences():
    candidate = np.array([8.0, 21.0, 34.0, 44.0])
    reference = np.array([10.0, 20.0, 30.0, 40.0])
    original_candidate = candidate.copy()
    original_reference = reference.copy()

    result = paired_mean_difference(candidate, reference, n_resamples=1001, seed=19)

    assert result["mean_difference"] == pytest.approx(1.75)
    assert result["per_seed_differences"] == [-2.0, 1.0, 4.0, 4.0]
    assert result["observed_zero_variance"] is False
    assert -2.0 <= result["ci95_low"] < result["ci95_high"] <= 4.0
    np.testing.assert_array_equal(candidate, original_candidate)
    np.testing.assert_array_equal(reference, original_reference)
    json.dumps(result, allow_nan=False)


def test_two_pair_bootstrap_resamples_observations_not_aggregate_means():
    result = paired_mean_difference([10.0, 24.0], [10.0, 20.0], seed=31)

    assert result["mean_difference"] == 2.0
    assert result["ci95_low"] == 0.0
    assert result["ci95_high"] == 4.0
    assert result["observed_zero_variance"] is False


def test_bootstrap_is_reproducible_and_reverses_with_difference_direction():
    candidate = [2.0, 7.0, 4.0, 15.0, 8.0]
    reference = [3.0, 4.0, 8.0, 6.0, 7.0]
    options = {"n_resamples": 1013, "seed": 81}

    result = paired_mean_difference(candidate, reference, **options)
    repeated = paired_mean_difference(candidate, reference, **options)
    reverse = paired_mean_difference(reference, candidate, **options)

    assert result == repeated
    assert reverse["mean_difference"] == pytest.approx(-result["mean_difference"])
    assert reverse["ci95_low"] == pytest.approx(-result["ci95_high"])
    assert reverse["ci95_high"] == pytest.approx(-result["ci95_low"])
    np.testing.assert_array_equal(
        reverse["per_seed_differences"], -np.array(result["per_seed_differences"])
    )


def test_bootstrap_does_not_consume_global_numpy_random_state():
    before = np.random.get_state()

    paired_mean_difference([2.0, 7.0, 5.0], [1.0, 2.0, 4.0], n_resamples=71, seed=3)

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_bootstrap_seed_and_resample_count_do_not_change_point_estimate():
    first = paired_mean_difference([1, 3, 8, 19], [2, 4, 7, 9], n_resamples=31, seed=4)
    second = paired_mean_difference([1, 3, 8, 19], [2, 4, 7, 9], n_resamples=57, seed=19)

    for field in ("n_pairs", "mean_difference", "per_seed_differences"):
        assert first[field] == second[field]


def test_bootstrap_batches_preserve_the_same_resampling_sequence():
    differences = np.linspace(-3.0, 5.0, 1001)
    n_resamples = 1001
    seed = 27
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(n_resamples, differences.size))
    reference_means = differences[indices].mean(axis=1)
    expected_low, expected_high = np.quantile(reference_means, [0.025, 0.975])

    result = paired_mean_difference(
        differences, np.zeros_like(differences), n_resamples=n_resamples, seed=seed
    )

    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["ci95_low"] == pytest.approx(expected_low, abs=1e-14)
    assert result["ci95_high"] == pytest.approx(expected_high, abs=1e-14)


def test_large_finite_observations_do_not_overflow_the_mean():
    result = paired_mean_difference([1e308, 8e307], [0.0, 0.0], n_resamples=31, seed=4)

    assert result["mean_difference"] == pytest.approx(9e307)
    assert np.isfinite(result["ci95_low"])
    assert np.isfinite(result["ci95_high"])


def test_statistics_reject_overflowing_pair_differences():
    with pytest.raises(ValueError, match="finite"):
        paired_mean_difference([1e308, 1.0], [-1e308, 0.0])


def test_extreme_quantile_interpolation_preserves_finite_bounds():
    result = paired_mean_difference([1e308, -1e308], [0.0, 0.0], n_resamples=2, seed=17)

    assert result["mean_difference"] == 0.0
    assert result["ci95_low"] == pytest.approx(-9.5e307)
    assert result["ci95_high"] == pytest.approx(9.5e307)


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        ([], []),
        ([1.0], [2.0]),
        ([1.0, 2.0], [3.0]),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        (1.0, [1.0, 2.0]),
        ([1.0, 2.0], 1.0),
        ([[1.0, 2.0]], [[1.0, 2.0]]),
        ([[1.0], [2.0]], [1.0, 2.0]),
        ([1.0, 2.0], [[1.0], [2.0]]),
        ([float("nan"), 1.0], [1.0, 2.0]),
        ([1.0, 2.0], [float("nan"), 1.0]),
        ([float("inf"), 1.0], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, -float("inf")]),
    ],
)
def test_statistics_reject_invalid_pair_arrays(candidate, reference):
    with pytest.raises((TypeError, ValueError)):
        paired_mean_difference(candidate, reference)


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, "10", True, False, np.bool_(True)])
def test_statistics_reject_invalid_resample_count(n_resamples):
    with pytest.raises((TypeError, ValueError)):
        paired_mean_difference([1.0, 2.0], [2.0, 3.0], n_resamples=n_resamples)


@pytest.mark.parametrize("seed", [-1, 0.5, "2", True, False, np.bool_(False)])
def test_statistics_reject_invalid_seed(seed):
    with pytest.raises((TypeError, ValueError)):
        paired_mean_difference([1.0, 2.0], [2.0, 3.0], seed=seed)


def test_statistics_accept_numpy_integer_parameters_and_single_resample():
    result = paired_mean_difference(
        [1.0, 4.0], [1.0, 2.0], n_resamples=np.int64(1), seed=np.int64(0)
    )

    assert result["n_pairs"] == 2
    assert result["mean_difference"] == 1.0
    assert np.isfinite(result["ci95_low"])
    assert result["ci95_low"] == result["ci95_high"]
