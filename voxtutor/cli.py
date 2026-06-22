"""Command-line interface. Prints ASCII only (Windows consoles are cp1252)."""

from __future__ import annotations

import argparse

from voxtutor.config import Settings
from voxtutor.runner import run
from voxtutor.scorers import ALIGNMENT, NORMALIZED, SCORERS
from voxtutor.synth import REGIMES
from voxtutor.types import Config


def _build_config(args: argparse.Namespace) -> Config:
    s = Settings.from_env()
    seed = getattr(args, "seed", None)
    samples = getattr(args, "samples", None)
    return Config(
        method=getattr(args, "method", None) or s.method,
        regime=getattr(args, "regime", None) or s.regime,
        labels=getattr(args, "labels", None) or s.labels,
        samples=samples if samples is not None else s.samples,
        seed=seed if seed is not None else s.seed,
        backend=getattr(args, "backend", None) or s.backend,
    )


def _ingredients(name: str) -> str:
    if name == "random":
        return "baseline"
    parts = []
    parts.append("align" if name in ALIGNMENT else "fixed")
    parts.append("gop-norm" if name in NORMALIZED else "raw")
    return "+".join(parts)


def _cmd_score(args: argparse.Namespace) -> None:
    cfg = _build_config(args)
    r = run(cfg)
    n_pos = int(r.labels.sum())
    print(f"method   : {cfg.method}  ({_ingredients(cfg.method)})")
    print(f"regime   : {cfg.regime}   labels={cfg.labels}  utterances={cfg.samples}  "
          f"seed={cfg.seed}")
    print(f"positions: {len(r.labels)} scored, {n_pos} mispronounced (ground truth)")
    print(f"auroc    : {r.auroc:.4f}   (1.0 = mispronunciations fully recovered, "
          "0.5 = chance)")


def _cmd_compare(args: argparse.Namespace) -> None:
    print(f"regime {args.regime}  utterances={args.samples}  seed={args.seed}  "
          f"labels={args.labels}")
    print(f"{'scorer':14s} {'ingredients':16s} {'auroc':>8s}")
    print("-" * 42)
    for name in SCORERS:
        cfg = Config(name, args.regime, args.labels, args.samples, args.seed, "numpy")
        r = run(cfg)
        print(f"{name:14s} {_ingredients(name):16s} {r.auroc:8.4f}")


def _cmd_regimes(_args: argparse.Namespace) -> None:
    blurbs = {
        "clean": "control: fixed rate, base noise -- every scorer should recover",
        "warped": "speaking-rate variation (breaks fixed segmentation)",
        "noisy": "heteroscedastic channel noise (breaks raw distance)",
    }
    print(f"{'regime':10s} description")
    print("-" * 70)
    for name in REGIMES:
        print(f"{name:10s} {blurbs.get(name, '')}")


def _cmd_eval(_args: argparse.Namespace) -> None:
    from evals.harness import main as harness_main

    harness_main()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="voxtutor", description="Pronunciation-assessment benchmark.")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("score", help="run one scorer on one regime")
    ps.add_argument("--method", choices=list(SCORERS))
    ps.add_argument("--regime", choices=list(REGIMES))
    ps.add_argument("--labels", choices=["real", "scrambled"])
    ps.add_argument("--samples", type=int)
    ps.add_argument("--seed", type=int)
    ps.add_argument("--backend", choices=["numpy", "scipy"])
    ps.set_defaults(func=_cmd_score)

    pc = sub.add_parser("compare", help="compare all scorers on a regime")
    pc.add_argument("--regime", default="warped", choices=list(REGIMES))
    pc.add_argument("--labels", default="real", choices=["real", "scrambled"])
    pc.add_argument("--samples", type=int, default=40)
    pc.add_argument("--seed", type=int, default=0)
    pc.set_defaults(func=_cmd_compare)

    pr = sub.add_parser("regimes", help="list the distortion regimes")
    pr.set_defaults(func=_cmd_regimes)

    pe = sub.add_parser("eval", help="run the full offline benchmark + write RESULTS.md")
    pe.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
