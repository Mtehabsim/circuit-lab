from __future__ import annotations

import argparse
from pathlib import Path

from .model import build_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/models/previous-token.pt")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_checkpoint(str(output))
    print(f"Wrote deterministic challenge checkpoint: {output}")


if __name__ == "__main__":
    main()
