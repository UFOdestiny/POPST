"""Run an experiment or suite directly from its YAML configuration."""

import argparse
import json
from pathlib import Path

from config.loader import load_config, read_yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Run or suite YAML")
    parser.add_argument("--config", type=Path, help="Alias for the positional YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Preview without execution")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", help="Single-run device override: cpu, cuda, or cuda:N")
    parser.add_argument("--run", type=Path, help="Exact run directory supplying best.pt")
    args = parser.parse_args(argv)
    if args.path and args.config:
        parser.error("Specify the YAML path once, positionally or with --config")
    path = args.path or args.config
    if args.run:
        source = args.run.resolve()
        path = path or source / "config.requested.yaml"
    if path is None:
        parser.error("A run/suite YAML or --run directory is required")

    if "runs" in read_yaml(path):
        if args.set or args.device or args.run:
            parser.error("Configure suite matrix, overrides, and devices in its YAML; --set, --device, and --run apply to single runs")
        from engine.suite import run_suite

        return run_suite(path, dry_run=args.dry_run)

    overrides = list(args.set)
    if args.run:
        overrides.append(f"runtime.checkpoint={json.dumps(str(source / 'best.pt'))}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    config = load_config(path, overrides)
    if args.dry_run:
        import yaml

        print(yaml.safe_dump(config.to_dict(), sort_keys=False))
        return 0
    if (
        config.runtime.mode == "calibrate"
        and config.calibration.mode == "no"
    ):
        parser.error("Calibration requires calibration.mode=horizon/global")
    from engine.runner import run_experiment

    result = run_experiment(config)
    print(json.dumps({
        "run": str(result.directory),
        "checkpoint": str(result.checkpoint) if result.checkpoint else None,
        "metrics": result.metrics,
        "skipped": getattr(result, "skipped", False),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
