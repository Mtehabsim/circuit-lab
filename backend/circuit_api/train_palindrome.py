"""Reproducibly train the independently authored Palindrome Machine checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from .palindrome import train_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="var/models/palindrome-trained-v3.pt")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--examples", type=int, default=10_000)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    print(train_checkpoint(str(output), seed=args.seed, epochs=args.epochs, examples=args.examples))
    print(f"Saved independently trained checkpoint and provenance to {output}")


if __name__ == "__main__":
    main()
