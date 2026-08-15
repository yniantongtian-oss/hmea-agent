import numpy as np
import pytest

from hmea import ChainMDP, NoModulation, QAgent


def test_train_logs_final_step_and_per_seed_metrics():
    log = QAgent(ChainMDP(n_states=3, sigma_r=0.0), NoModulation(), seed=3).train(
        n_steps=11, n_seeds=4, log_every=5
    )

    np.testing.assert_array_equal(log["steps"], [5, 10, 11])
    assert log["bias"].shape == (3,)
    assert log["bias_by_seed"].shape == (3, 4)
    assert log["rmse_by_seed"].shape == (3, 4)
    assert log["q_start_by_seed"].shape == (3, 4)
    assert log["q_values"].shape == (4, 3, 2)
    assert log["audit"]["name"] == "none"


def test_training_is_reproducible():
    def train():
        return QAgent(ChainMDP(sigma_r=0.1), NoModulation(), seed=19).train(
            n_steps=300, n_seeds=6, log_every=70
        )

    first = train()
    second = train()
    for key in ("steps", "bias_by_seed", "rmse_by_seed", "q_values"):
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("alpha", 0.0),
        ("eps0", 1.1),
        ("eps_min", -0.1),
        ("eps_decay", 0.0),
        ("seed", -1),
    ],
)
def test_agent_rejects_invalid_parameters(keyword, value):
    with pytest.raises(ValueError):
        QAgent(ChainMDP(), NoModulation(), **{keyword: value})


@pytest.mark.parametrize("keyword", ["n_steps", "n_seeds", "log_every"])
def test_train_rejects_non_positive_sizes(keyword):
    arguments = {"n_steps": 10, "n_seeds": 2, "log_every": 5, keyword: 0}
    with pytest.raises(ValueError):
        QAgent(ChainMDP(), NoModulation()).train(**arguments)


def test_train_rejects_fractional_steps():
    with pytest.raises(TypeError):
        QAgent(ChainMDP(), NoModulation()).train(3.5)
