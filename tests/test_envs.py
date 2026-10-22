import numpy as np
import pytest

from hmea import ChainMDP
from hmea.envs import _NOISE_BLOCK


def test_deterministic_transitions_and_terminal_reward():
    env = ChainMDP(n_states=4, sigma_r=0.0)
    next_states, rewards, done = env.step(np.array([0, 1, 2, 3]), np.array([0, 1, 1, 0]))
    np.testing.assert_array_equal(next_states, [0, 2, 3, 3])
    np.testing.assert_array_equal(rewards, [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(done, [False, False, True, True])


def test_optimal_q_matches_small_chain():
    q_star = ChainMDP(n_states=3, gamma=0.9, sigma_r=0.0).optimal_q()
    np.testing.assert_allclose(q_star, [[0.81, 0.9], [0.81, 1.0], [0.0, 0.0]])


def test_noise_stream_matches_the_buffered_generator():
    states = np.array([0, 1, 1, 0])
    actions = np.array([0, 0, 1, 1])
    env = ChainMDP(n_states=4, sigma_r=0.2)
    env.rng = np.random.default_rng(42)

    first = env.step(states, actions)[1]
    second = env.step(states, actions)[1]

    np.testing.assert_allclose(
        first,
        [0.06094341595088627, -0.20799682124809912, 0.15009023916129147, 0.1881129432782428],
    )
    np.testing.assert_allclose(
        second,
        [-0.3902070377307673, -0.26043590137246364, 0.025568080633457075, -0.06324851846871644],
    )


def test_noise_batch_larger_than_the_internal_block_is_complete():
    count = _NOISE_BLOCK + 3
    env = ChainMDP(n_states=4, sigma_r=0.2)
    env.rng = np.random.default_rng(7)
    states = np.zeros(count, dtype=np.int64)
    actions = np.zeros(count, dtype=np.int64)

    rewards = env.step(states, actions)[1]
    again = env.step(states[:4], actions[:4])[1]

    assert rewards.shape == (count,)
    assert np.isfinite(rewards).all()
    assert np.count_nonzero(rewards) == count
    assert again.shape == (4,)
    assert np.isfinite(again).all()


def test_environment_rejects_boolean_counts():
    with pytest.raises(TypeError, match="n_states"):
        ChainMDP(n_states=True)
    with pytest.raises(TypeError, match="n_seeds"):
        ChainMDP().reset(True)


def test_noise_is_reproducible_with_explicit_generator():
    states = np.array([0, 1, 1, 0])
    actions = np.array([0, 0, 1, 1])
    outputs = []
    for _ in range(2):
        env = ChainMDP(n_states=4, sigma_r=0.2)
        env.rng = np.random.default_rng(42)
        outputs.append(env.step(states, actions)[1])
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_potential_shaping_is_added_to_rewards():
    gamma = 0.9
    potential = np.array([0.0, 0.4, 0.7, 0.0])

    def shaping(state, next_state):
        return gamma * potential[next_state] - potential[state]

    env = ChainMDP(n_states=4, gamma=gamma, sigma_r=0.0, shaping=shaping)
    _, rewards, _ = env.step(np.array([0, 2]), np.array([1, 1]))
    np.testing.assert_allclose(rewards, [0.36, 0.3])
    assert np.isfinite(env.optimal_q()).all()


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("n_states", 1, ValueError),
        ("n_states", 2.5, TypeError),
        ("gamma", 1.0, ValueError),
        ("gamma", -0.1, ValueError),
        ("sigma_r", -0.1, ValueError),
        ("shaping", 3, TypeError),
    ],
)
def test_environment_rejects_invalid_parameters(keyword, value, error):
    with pytest.raises(error):
        ChainMDP(**{keyword: value})


@pytest.mark.parametrize(
    ("states", "actions", "error"),
    [
        (np.array([[0]]), np.array([0]), ValueError),
        (np.array([0, 1]), np.array([0]), ValueError),
        (np.array([], dtype=int), np.array([], dtype=int), ValueError),
        (np.array([0.0]), np.array([0]), TypeError),
        (np.array([0]), np.array([0.0]), TypeError),
        (np.array([99]), np.array([0]), ValueError),
        (np.array([0]), np.array([2]), ValueError),
    ],
)
def test_step_validates_inputs(states, actions, error):
    with pytest.raises(error):
        ChainMDP().step(states, actions)
