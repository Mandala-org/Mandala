"""Run one of a fixed number of restartable, budgeted training segments.

This runner is model/dataset independent. It invokes the normal shared trainer,
preserves optimizer/scheduler/RNG restart state, and evaluates the selected best
checkpoint only after the final preregistered segment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--segment", type=int, required=True)
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=45 * 3600)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--evaluate-final-test", action="store_true")
    args = parser.parse_args(argv)
    if (
        not 1 <= args.segment <= args.segments
        or not 0 < args.budget_seconds <= 48 * 3600
    ):
        parser.error("Invalid segment or segment budget")
    args.records.mkdir(parents=True, exist_ok=True)
    with (args.records / ".segment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = args.records / f"segment_{args.segment}.json"
        if record_path.exists():
            raise FileExistsError(
                f"Segment already attempted; inspect its record: {record_path}"
            )
        if args.segment > 1:
            previous = json.loads(
                (args.records / f"segment_{args.segment-1}.json").read_text()
            )
            if previous["status"] != "completed":
                raise RuntimeError("Previous segment failed; inspect before continuing")
        record = {
            "segment": args.segment,
            "segments": args.segments,
            "budget_seconds": args.budget_seconds,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        write_json(record_path, record)
        started = time.perf_counter()
        try:
            import torch
            from omegaconf import OmegaConf
            from scripts.train import main as train

            latest = args.output / "latest_checkpoint.pt"
            config = OmegaConf.load(args.config)
            max_epochs = config.get("training", {}).get(
                "max_epochs", config.get("max_epochs", 4000)
            )
            if args.segment == 1 and latest.exists():
                raise FileExistsError("Fresh segment refuses an existing checkpoint")
            if args.segment > 1 and not latest.is_file():
                raise FileNotFoundError("Continuation requires a checkpoint")
            checkpoint = (
                torch.load(latest, map_location="cpu", weights_only=False)
                if latest.is_file()
                else None
            )
            if checkpoint and not checkpoint.get("epoch_boundary_resume", {}).get(
                "safe_to_resume"
            ):
                raise ValueError("Checkpoint is not at a safe epoch boundary")
            if checkpoint is None or int(checkpoint["epoch"]) < max_epochs:
                cli = [
                    "--config",
                    str(args.config),
                    "--output",
                    str(args.output),
                    "--accelerator",
                    args.accelerator,
                    "--max-wall-clock-seconds",
                    str(args.budget_seconds),
                    "--wall-clock-scope",
                    "session",
                    "--log-dir",
                    str(args.records / "metrics"),
                ]
                if latest.exists():
                    cli.extend(["--resume", str(latest)])
                record["last_metrics"] = train(cli)
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            if not checkpoint.get("epoch_boundary_resume", {}).get("safe_to_resume"):
                raise ValueError("Training did not save a safe epoch boundary")
            record["completed_epochs"] = int(checkpoint["epoch"])
            record["learning_rates"] = [
                group["lr"]
                for optimizer in checkpoint["optimizer_states"]
                for group in optimizer["param_groups"]
            ]
            record["checkpoint"] = str(latest)
            if args.segment == args.segments:
                from scripts.evaluate_operator_checkpoint import evaluate

                best = args.output / "best_model.pt"
                splits = ["val", "test"] if args.evaluate_final_test else ["val"]
                for split in splits:
                    path = args.records / f"best_{split}_evaluation.json"
                    if path.exists():
                        raise FileExistsError(
                            f"Final evaluation already exists: {path}"
                        )
                    write_json(
                        path,
                        evaluate(
                            args.config, best, split=split, accelerator=args.accelerator
                        ),
                    )
            record["status"] = "completed"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = time.perf_counter() - started
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(record_path, record)
            print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
