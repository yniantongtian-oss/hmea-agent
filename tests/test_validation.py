import numpy as np
import pytest

from hmea import (
    LaggedModulation,
    NoModulation,
    OnlineModulation,
    audit,
    check_bounded,
    check_current_error_independence,
    validate_modulator,
)


def test_audit_is_descriptive_for_built_in_policies():
    baseline = audit(NoModulation())
    online = audit(OnlineModulation())
    lagged_modulator = LaggedModulation(lag=4)
    lagged = audit(lagged_modulator)

    assert baseline == {
        "name": "none",
        "bounded": True,
        "lag": 0,
        "uses_current_td_error": False,
        "current_error_independent": True,
    }
    assert online["uses_current_td_error"] is True
    assert lagged["lag"] == 4
    assert check_bounded(lagged_modulator)
    assert check_current_error_independence(lagged_modulator)


def test_validate_accepts_custom_structural_policy():
    class Custom:
        name = "custom"
        bounds = (0.5, 1.5)
        lag = 0
        uses_current_td_error = False

        def reset(self, n_seeds):
            self.n_seeds = n_seeds

        def multiplier(self, td_error, step):
            return np.ones_like(td_error)

    assert validate_modulator(Custom())["bounded"] is True


@pytest.mark.parametrize(
    "candidate",
    [
        object(),
        type(
            "BadBounds",
            (),
            {
                "name": "bad",
                "bounds": (0.0, np.inf),
                "lag": 0,
                "reset": lambda self, n: None,
                "multiplier": lambda self, error, step: error,
            },
        )(),
        type(
            "BadLag",
            (),
            {
                "name": "bad",
                "bounds": (1.0, 1.0),
                "lag": -1,
                "reset": lambda self, n: None,
                "multiplier": lambda self, error, step: error,
            },
        )(),
        type(
            "FractionalLag",
            (),
            {
                "name": "bad",
                "bounds": (1.0, 1.0),
                "lag": 1.5,
                "uses_current_td_error": False,
                "reset": lambda self, n: None,
                "multiplier": lambda self, error, step: error,
            },
        )(),
        type(
            "BadDeclaration",
            (),
            {
                "name": "bad",
                "bounds": (1.0, 1.0),
                "lag": 0,
                "uses_current_td_error": "no",
                "reset": lambda self, n: None,
                "multiplier": lambda self, error, step: error,
            },
        )(),
    ],
)
def test_validate_rejects_invalid_policy(candidate):
    with pytest.raises(ValueError):
        validate_modulator(candidate)
