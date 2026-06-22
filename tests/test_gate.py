"""The eval gate must pass: every dissociation check holds."""

from evals.gate import run_checks


def test_gate_all_pass():
    checks = run_checks()
    failures = [msg for ok, msg in checks if not ok]
    assert not failures, "gate failures:\n" + "\n".join(failures)
    assert len(checks) >= 20  # the gate asserts a battery of conditions
