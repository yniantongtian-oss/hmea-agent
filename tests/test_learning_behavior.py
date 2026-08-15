from hmea import ChainMDP, LaggedModulation, OnlineModulation, QAgent


def test_immediate_and_lagged_policies_separate_in_reference_scenario():
    outputs = {}
    for name, modulator in (
        ("immediate", OnlineModulation(beta=1.5, tau=0.5, bounds=(0.5, 2.0))),
        ("lagged", LaggedModulation(lag=8, beta=1.5, tau=0.5, bounds=(0.5, 2.0))),
    ):
        outputs[name] = QAgent(ChainMDP(), modulator, seed=0).train(
            n_steps=20_000, n_seeds=24, log_every=1_000
        )

    immediate = outputs["immediate"]["bias"][-1]
    lagged = outputs["lagged"]["bias"][-1]
    assert immediate > lagged + 0.25
    assert outputs["immediate"]["rmse"][-1] > outputs["lagged"]["rmse"][-1]
