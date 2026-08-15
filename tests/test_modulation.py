import numpy as np
import pytest

from hmea import (
    ClippedModulation,
    HomeostaticModulation,
    LaggedModulation,
    Modulator,
    NoModulation,
    OnlineModulation,
)


def test_no_modulation_returns_reusable_ones():
    modulator = NoModulation()
    first = modulator.multiplier(np.array([-1.0, 2.0]), 0)
    second = modulator.multiplier(np.array([5.0, 6.0]), 1)
    np.testing.assert_array_equal(first, [1.0, 1.0])
    assert first is second
    assert isinstance(modulator, Modulator)


def test_modulator_reset_requires_an_integer_seed_count():
    with pytest.raises(TypeError):
        NoModulation().reset(1.5)


def test_online_modulation_is_monotone_and_bounded():
    modulator = OnlineModulation(beta=1.5, tau=0.5, bounds=(0.5, 2.0))
    values = modulator.multiplier(np.array([-10.0, 0.0, 10.0]), 0).copy()
    assert np.all(np.diff(values) > 0)
    np.testing.assert_allclose(values, [0.5, 1.0, 2.0])


def test_clipped_modulation_respects_narrow_bounds():
    values = ClippedModulation().multiplier(np.array([-10.0, 10.0]), 0)
    np.testing.assert_allclose(values, [0.8, 1.2])


def test_lagged_modulation_uses_fixed_delay():
    modulator = LaggedModulation(lag=2, beta=0.5, tau=1.0, bounds=(0.5, 1.5))
    modulator.reset(1)
    np.testing.assert_allclose(modulator.multiplier(np.array([1.0]), 0), [1.0])
    np.testing.assert_allclose(modulator.multiplier(np.array([-1.0]), 1), [1.0])
    expected = 1.0 + 0.5 * np.tanh(1.0)
    np.testing.assert_allclose(modulator.multiplier(np.array([100.0]), 2), [expected])


def test_homeostatic_multiplier_uses_previous_state():
    left = HomeostaticModulation(eta=0.5)
    right = HomeostaticModulation(eta=0.5)
    left.reset(1)
    right.reset(1)
    np.testing.assert_array_equal(left.multiplier(np.array([2.0]), 0), [1.0])
    np.testing.assert_array_equal(right.multiplier(np.array([2.0]), 0), [1.0])
    np.testing.assert_allclose(
        left.multiplier(np.array([100.0]), 1),
        right.multiplier(np.array([-100.0]), 1),
    )
    assert not np.allclose(
        left.multiplier(np.array([0.0]), 2),
        right.multiplier(np.array([0.0]), 2),
    )


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: OnlineModulation(beta=0.0), ValueError),
        (lambda: OnlineModulation(tau=0.0), ValueError),
        (lambda: OnlineModulation(bounds=(0.0, 1.0)), ValueError),
        (lambda: OnlineModulation(bounds=(2.0, 1.0)), ValueError),
        (lambda: LaggedModulation(lag=0), ValueError),
        (lambda: LaggedModulation(lag=1.5), ValueError),
    ],
)
def test_modulators_reject_invalid_parameters(factory, error):
    with pytest.raises(error):
        factory()


def test_homeostatic_modulator_rejects_shape_change():
    modulator = HomeostaticModulation()
    modulator.reset(2)
    with pytest.raises(ValueError):
        modulator.multiplier(np.array([1.0]), 0)
