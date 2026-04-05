from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

from .config import load_config
from .export import export_checkpoint, verify_export
from .metrics import generate_run_plots, load_metrics_run, render_summary_text, summarize_run
from .native import inspect_binpack
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

    metrics_parser = subparsers.add_parser("metrics", help="Summarize a training run and generate plots")
    metrics_parser.add_argument("--run-dir", required=True)

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

    if args.command == "metrics":
        run = load_metrics_run(args.run_dir)
        summary = summarize_run(run)
        plots = generate_run_plots(run)
        output = {
            "summary": summary,
            "plots": [str(path) for path in plots],
            "text": render_summary_text(summary),
        }
        print(output["text"])
        return

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
