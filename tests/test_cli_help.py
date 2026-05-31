"""Smoke tests: every CLI must respond to --help without importing heavy deps eagerly."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "cv_diffusion.cli.preprocess",
        "cv_diffusion.cli.train_lora",
        "cv_diffusion.cli.generate",
        "cv_diffusion.cli.train_classifier",
        "cv_diffusion.cli.evaluate",
    ],
)
def test_cli_help(module_name, capsys):
    module = importlib.import_module(module_name)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower()
