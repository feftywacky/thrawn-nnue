from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import unittest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("must be >= 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError("must be >= 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError("must be >= 0")
    return parsed


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_test_suite(*, pattern: str, verbosity: int, failfast: bool) -> int:
    root = _project_root()
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(root / "tests"),
        pattern=pattern,
    )
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> None:
    parser = ArgumentParser(prog="thrawn-nnue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from a TOML config")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--console-mode", choices=["progress", "text"])
    train_parser.add_argument(
        "--init-checkpoint",
        help="Warm-start model weights from a checkpoint while using the supplied config and a fresh optimizer",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume from a checkpoint")
    resume_parser.add_argument("--checkpoint", required=True)
    resume_parser.add_argument("--console-mode", choices=["progress", "text"])

    export_parser = subparsers.add_parser("export", help="Export a checkpoint to .nnue")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--out", required=True)

    verify_parser = subparsers.add_parser("verify-export", help="Compare checkpoint and exported .nnue outputs")
    verify_parser.add_argument("--checkpoint", required=True)
    verify_parser.add_argument("--nnue", required=True)
    verify_parser.add_argument("--fen", action="append", default=[])

    inspect_parser = subparsers.add_parser("inspect-binpack", help="Inspect a .binpack dataset")
    inspect_parser.add_argument("--path", required=True)
    inspect_parser.add_argument("--skip-capture-positions", action="store_true")
    inspect_parser.add_argument("--skip-wdl-score-mismatch", action="store_true")
    inspect_parser.add_argument("--max-abs-score", type=_non_negative_float, default=0.0)
    inspect_parser.add_argument(
        "--sample-entries",
        type=_non_negative_int,
        default=0,
        help="Stop after this many decoded entries; 0 scans the full file",
    )

    inspect_all_parser = subparsers.add_parser(
        "inspect-binpack-dir",
        help="Inspect all .binpack files under a directory and aggregate the results",
    )
    inspect_all_parser.add_argument("--path", required=True)
    inspect_all_parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=None,
        help="Number of .binpack files to inspect concurrently; defaults to available CPUs capped by file count",
    )
    inspect_all_parser.add_argument("--skip-capture-positions", action="store_true")
    inspect_all_parser.add_argument("--skip-wdl-score-mismatch", action="store_true")
    inspect_all_parser.add_argument("--max-abs-score", type=_non_negative_float, default=0.0)
    inspect_all_parser.add_argument(
        "--sample-entries",
        type=_non_negative_int,
        default=0,
        help="Stop after this many decoded entries per file; 0 scans every file fully",
    )

    metrics_parser = subparsers.add_parser("metrics", help="Summarize a training run and generate plots")
    metrics_parser.add_argument("--run-dir", required=True)
    metrics_parser.add_argument("--json", action="store_true")

    test_parser = subparsers.add_parser("test", help="Run the unittest suite")
    test_parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Glob pattern for unittest discovery under tests/ (default: test_*.py)",
    )
    test_parser.add_argument("--failfast", action="store_true")
    test_parser.add_argument(
        "--verbosity",
        type=_positive_int,
        default=2,
        help="unittest verbosity level (default: 2)",
    )

    args = parser.parse_args()

    if args.command == "train":
        from .config import load_config
        from .training import train_from_config

        config = load_config(args.config)
        checkpoint_path = train_from_config(
            config,
            console_mode=args.console_mode,
            init_checkpoint=args.init_checkpoint,
        )
        print(str(checkpoint_path))
        return

    if args.command == "resume":
        from .training import resume_training

        checkpoint_path = resume_training(args.checkpoint, console_mode=args.console_mode)
        print(str(checkpoint_path))
        return

    if args.command == "export":
        from .export import export_checkpoint

        output = export_checkpoint(args.checkpoint, args.out)
        print(str(output))
        return

    if args.command == "verify-export":
        from .export import verify_export

        results = verify_export(args.checkpoint, args.nnue, args.fen or None)
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if args.command == "inspect-binpack":
        from .native import inspect_binpack

        stats = inspect_binpack(
            args.path,
            skip_capture_positions=args.skip_capture_positions,
            skip_wdl_score_mismatch=args.skip_wdl_score_mismatch,
            max_abs_score=args.max_abs_score,
            sample_entries=args.sample_entries,
        )
        print(json.dumps(stats, indent=2))
        return

    if args.command == "inspect-binpack-dir":
        from .native import discover_binpack_files, inspect_binpack_collection

        paths = discover_binpack_files(args.path)
        stats = inspect_binpack_collection(
            paths,
            jobs=args.jobs,
            skip_capture_positions=args.skip_capture_positions,
            skip_wdl_score_mismatch=args.skip_wdl_score_mismatch,
            max_abs_score=args.max_abs_score,
            sample_entries=args.sample_entries,
        )
        print(json.dumps(stats, indent=2))
        return

    if args.command == "metrics":
        from .metrics import generate_run_plots, load_metrics_run, render_summary_text, summarize_run

        run = load_metrics_run(args.run_dir)
        summary = summarize_run(run)
        plots = generate_run_plots(run)
        output = {
            "summary": summary,
            "plots": [str(path) for path in plots],
            "text": render_summary_text(summary),
        }
        if args.json:
            print(json.dumps(output, indent=2, sort_keys=True))
        else:
            print(output["text"])
        return

    if args.command == "test":
        raise SystemExit(_run_test_suite(pattern=args.pattern, verbosity=args.verbosity, failfast=args.failfast))

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
