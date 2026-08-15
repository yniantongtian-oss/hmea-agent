import numpy as np
import pytest

from hmea import HomeostaticState


def test_state_tracks_error_deviation_from_setpoint():
    state = HomeostaticState(setpoint=1.0, eta=0.5)
    np.testing.assert_allclose(state.update(np.array([3.0, 1.0])), [1.0, 0.0])
    np.testing.assert_allclose(state.update(np.array([1.0, 1.0])), [0.5, 0.0])


def test_state_requires_initialization_before_read():
    with pytest.raises(RuntimeError):
        _ = HomeostaticState().value


def test_state_rejects_shape_change_and_non_finite_input():
    state = HomeostaticState()
    state.reset(2)
    with pytest.raises(ValueError):
        state.update(np.array([1.0]))
    with pytest.raises(ValueError):
        state.update(np.array([1.0, np.nan]))


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("setpoint", np.inf), ("eta", 0.0), ("eta", 1.1)],
)
def test_state_rejects_invalid_parameters(keyword, value):
    with pytest.raises(ValueError):
        HomeostaticState(**{keyword: value})


def test_reset_requires_at_least_one_seed():
    with pytest.raises(ValueError):
        HomeostaticState().reset(0)
    with pytest.raises(TypeError):
        HomeostaticState().reset(1.5)
