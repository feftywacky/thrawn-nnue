from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

from .calibration import calibrate_scale
from .config import load_config
from .export import export_checkpoint, verify_export
from .metrics import generate_run_plots, load_metrics_run, render_summary_text, summarize_run
from .native import discover_binpack_files, inspect_binpack, inspect_binpack_collection
from .training import resume_training, train_from_config


def main() -> None:
    parser = ArgumentParser(prog="thrawn-nnue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from a TOML config")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--console-mode", choices=["progress", "text"])

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

    inspect_all_parser = subparsers.add_parser(
        "inspect-binpack-dir",
        help="Inspect all .binpack files under a directory and aggregate the results",
    )
    inspect_all_parser.add_argument("--path", required=True)

    metrics_parser = subparsers.add_parser("metrics", help="Summarize a training run and generate plots")
    metrics_parser.add_argument("--run-dir", required=True)
    metrics_parser.add_argument("--json", action="store_true")

    calibrate_parser = subparsers.add_parser(
        "calibrate-scale",
        help="Fit engine centipawn scale from raw exported NNUE output against validation .binpack targets",
    )
    calibrate_parser.add_argument("--nnue", required=True)
    calibrate_parser.add_argument("--validation-path", required=True)
    calibrate_parser.add_argument("--max-positions", type=int, default=300000)
    calibrate_parser.add_argument("--batch-size", type=int, default=1024)
    calibrate_parser.add_argument("--threads", type=int, default=4)
    calibrate_parser.add_argument("--fit-window-cp", type=float, default=600.0)
    calibrate_parser.add_argument("--min-fit-positions", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "train":
        config = load_config(args.config)
        checkpoint_path = train_from_config(config, console_mode=args.console_mode)
        print(str(checkpoint_path))
        return

    if args.command == "resume":
        checkpoint_path = resume_training(args.checkpoint, console_mode=args.console_mode)
        print(str(checkpoint_path))
        return

    if args.command == "export":
        output = export_checkpoint(args.checkpoint, args.out)
        print(str(output))
        return

    if args.command == "verify-export":
        results = verify_export(args.checkpoint, args.nnue, args.fen or None)
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    if args.command == "inspect-binpack":
        stats = inspect_binpack(args.path)
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    if args.command == "inspect-binpack-dir":
        paths = discover_binpack_files(args.path)
        stats = inspect_binpack_collection(paths)
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    if args.command == "metrics":
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

    if args.command == "calibrate-scale":
        result = calibrate_scale(
            args.nnue,
            args.validation_path,
            max_positions=args.max_positions,
            batch_size=args.batch_size,
            threads=args.threads,
            fit_window_cp=args.fit_window_cp,
            min_fit_positions=args.min_fit_positions,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
