import pytest

from voxtutor.cli import main


def test_score_runs(capsys):
    main(["score", "--method", "gop", "--regime", "warped", "--samples", "8", "--seed", "0"])
    out = capsys.readouterr().out
    assert "auroc" in out and "gop" in out


def test_compare_lists_all_scorers(capsys):
    main(["compare", "--regime", "noisy", "--samples", "8", "--seed", "0"])
    out = capsys.readouterr().out
    for name in ("random", "naive", "aligned", "normalized", "gop"):
        assert name in out


def test_regimes_describes_each(capsys):
    main(["regimes"])
    out = capsys.readouterr().out
    assert "clean" in out and "warped" in out and "noisy" in out


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
