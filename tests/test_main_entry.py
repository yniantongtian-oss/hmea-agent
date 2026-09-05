import runpy
import sys

import pytest


def test_module_entry_point_delegates_to_cli(monkeypatch):
    calls: list[object] = []

    def fake_main(argv=None):
        calls.append(argv)
        return 0

    import hmea.cli as cli

    monkeypatch.setattr(cli, "main", fake_main)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("hmea", run_name="__main__")
    assert exc.value.code == 0
    assert len(calls) == 1
