import sys

import pytest
from conftest import make_policy

pytest.importorskip("msgpack")  # flux_action.serving.robolab imports the OpenPI codec

from flux_action import cli  # noqa: E402
from flux_action.serving import robolab  # noqa: E402


def test_setting_values_accept_bare_strings_and_json(monkeypatch, capsys):
    """``--setting sampler=euler`` is the documented SO-101 evaluation flag; only JSON scalars used to parse."""
    calls = []
    policy = make_policy()
    policy.serving_setup = {}

    def fake_load(checkpoint, **kwargs):
        calls.append(kwargs)
        return policy

    monkeypatch.setattr(robolab, "load_serving_policy", fake_load)
    monkeypatch.setattr(robolab, "serve", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flux-action",
            "serve-robolab",
            "--checkpoint",
            "x",
            "--setting",
            "sampler=euler",
            "--setting",
            "num_inference_steps=4",
            "--setting",
            "sampler_shift=6.93",
        ],
    )
    cli.main()
    assert calls[0]["settings"] == {
        "sampler": "euler",
        "num_inference_steps": 4,
        "sampler_shift": 6.93,
    }
    capsys.readouterr()


def test_setting_without_a_value_is_rejected():
    with pytest.raises(ValueError, match="is not KEY=VALUE"):
        cli._parse_settings(["sampler"])
