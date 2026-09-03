from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from mip_attack.config import load_config, verify_assets


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mip-attack",
        description="Craft and evaluate malicious image patches against multimodal OS agents.",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    craft_parser = subparsers.add_parser("craft", help="Optimize a patch")
    _common_parser(craft_parser)
    craft_parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max_optimizer_steps (useful for smoke tests)",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a crafted patch")
    _common_parser(evaluate_parser)
    evaluate_parser.add_argument("--patch-run", required=True, type=Path)
    evaluate_parser.add_argument("--patch-id")

    run_parser = subparsers.add_parser(
        "run", help="Craft and evaluate a complete paper experiment"
    )
    _common_parser(run_parser)
    run_parser.add_argument(
        "--max-steps",
        type=int,
        help="Override max_optimizer_steps (useful for smoke tests)",
    )

    summarize_parser = subparsers.add_parser(
        "summarize", help="Regenerate aggregate metrics from evaluation.csv"
    )
    summarize_parser.add_argument("--evaluation-run", required=True, type=Path)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate config, prompt splits, and screen assets"
    )
    validate_parser.add_argument("--config", required=True, type=Path)
    validate_parser.add_argument("--target-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env", override=False)
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.command == "summarize":
        from mip_attack.reporting import summarize_evaluation

        report = summarize_evaluation(args.evaluation_run / "evaluation.csv")
        print(json.dumps({key: str(value) for key, value in report.items()}, indent=2))
        return 0

    config = load_config(args.config, getattr(args, "target_file", None))
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed

    if args.command == "validate":
        print(json.dumps(verify_assets(config), indent=2, sort_keys=True))
        return 0

    from mip_attack.evaluation import evaluate
    from mip_attack.runner import craft, run_experiment

    if args.command in {"craft", "run"}:
        if args.max_steps is not None:
            if args.max_steps <= 0:
                raise ValueError("--max-steps must be positive")
            config["craft"]["max_optimizer_steps"] = args.max_steps
        output = (
            craft(config, run_id=args.run_id)
            if args.command == "craft"
            else run_experiment(config, run_id=args.run_id)
        )
    else:
        output = evaluate(
            config,
            patch_run=args.patch_run,
            run_id=args.run_id,
            patch_id=args.patch_id,
        )
    print(output)
    return 0
