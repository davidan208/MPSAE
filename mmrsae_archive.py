#!/usr/bin/env python3
"""Copy the durable part of completed MMRSAE fits into a compact archive.

Kept, in the same layout as the run directory (so mmrsae_reporting.py works on
the archive): every JSON/JSONL/TXT record and predictions.csv, and a weights-only
``final_model.pt`` extracted from ``last.pt`` (plus any ``best*.pt``). Dropped:
resume state (optimizer, RNG), linear-probe checkpoints, encoded caches and
other ``.npy`` arrays, all regenerable from the final weights.

    python mmrsae_archive.py --output-dir runs/mmrsae --archive-dir runs/mmrsae_archive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

BACKBONES = ("resnet50", "resnet18")
RECORD_SUFFIXES = {".json", ".jsonl", ".txt"}
RECORD_NAMES = {"predictions.csv"}
SKIPPED_DIRECTORIES = {"encoded_cache", "wandb"}
RESUME_ONLY = {"optimizer", "rng", "context_rng"}


def _records(directory):
    """Small result files below one backbone directory, never caches or arrays."""
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part in SKIPPED_DIRECTORIES for part in relative.parts[:-1]) or not path.is_file():
            continue
        if path.suffix in RECORD_SUFFIXES or path.name in RECORD_NAMES:
            yield relative


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _final_weights(checkpoint, target, checkpoint_sha256):
    """Write the model without resume state; skip when this checkpoint is already archived.

    The run evaluates last.pt after its final update and records its SHA-256 in
    summary.json, so the archived weights are exactly the ones behind the report.
    """
    import torch
    from repair_data import sha256_file
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=False)
        if checkpoint_sha256 and existing.get("source_checkpoint_sha256") == checkpoint_sha256:
            return False
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError(f"{checkpoint} differs from the checkpoint evaluated in summary.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {key: value for key, value in payload.items() if key not in RESUME_ONLY}
    weights["source_checkpoint_sha256"] = checkpoint_sha256
    temporary = target.with_name(target.name + ".partial")
    torch.save(weights, temporary)
    temporary.replace(target)
    return True


def archive(output_dir, archive_dir):
    output_dir, archive_dir = Path(output_dir), Path(archive_dir)
    archived = []
    for trial_path in sorted(output_dir.glob("*/trial.json")):
        trial = trial_path.parent
        destination = archive_dir / trial.name
        complete = []
        for backbone in BACKBONES:
            directory = trial / backbone
            try:
                summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not summary.get("complete"):
                continue
            for relative in _records(directory):
                _copy(directory / relative, destination / backbone / relative)
            if (directory / "last.pt").exists():
                _final_weights(directory / "last.pt", destination / backbone / "final_model.pt",
                               summary.get("checkpoint_sha256"))
            for best in sorted(directory.glob("best*.pt")):
                _copy(best, destination / backbone / best.name)
            complete.append(backbone)
        if complete:
            for name in ("trial.json", "configuration.json", "gpu_preflight.json"):
                if (trial / name).exists():
                    _copy(trial / name, destination / name)
            archived.append(f"{trial.name} ({', '.join(complete)})")
    reports = output_dir / "reports"
    if reports.is_dir():
        for path in sorted(reports.rglob("*")):
            if path.is_file():
                _copy(path, archive_dir / "reports" / path.relative_to(reports))
    return archived


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mmrsae"))
    parser.add_argument("--archive-dir", type=Path, help="Default: <output-dir>_archive")
    args = parser.parse_args(argv)
    archive_dir = args.archive_dir or args.output_dir.with_name(args.output_dir.name + "_archive")
    archived = archive(args.output_dir, archive_dir)
    print(f"MMRSAE archive: {archive_dir} ({len(archived)} trials with completed fits)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
