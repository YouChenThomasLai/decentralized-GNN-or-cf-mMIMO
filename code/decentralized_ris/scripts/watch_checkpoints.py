#!/usr/bin/env python3
"""Retain selected iterations from a training run's rolling last.pt."""

import argparse
import os
import shutil
import time
from pathlib import Path

import torch


def checkpoint_iteration(path):
    return int(torch.load(path, map_location="cpu", weights_only=False)["iteration"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("iterations", type=int, nargs="+")
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()

    checkpoint_dir = args.run_dir / "checkpoints"
    source = checkpoint_dir / "last.pt"
    pending = {
        iteration
        for iteration in args.iterations
        if not (checkpoint_dir / f"iter{iteration}.pt").exists()
    }
    while pending:
        try:
            current = checkpoint_iteration(source)
        except (FileNotFoundError, OSError, RuntimeError, EOFError):
            time.sleep(args.poll_seconds)
            continue
        if current in pending:
            destination = checkpoint_dir / f"iter{current}.pt"
            temporary = destination.with_suffix(".pt.tmp")
            shutil.copy2(source, temporary)
            if checkpoint_iteration(temporary) == current:
                os.replace(temporary, destination)
                pending.remove(current)
                print(f"saved {destination}", flush=True)
            else:
                temporary.unlink(missing_ok=True)
        if pending and current > min(pending):
            raise SystemExit(f"missed checkpoint iteration {min(pending)}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
