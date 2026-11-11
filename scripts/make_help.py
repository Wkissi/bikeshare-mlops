"""Affiche les cibles du Makefile documentées par un commentaire « ## »."""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET_PATTERN = re.compile(r"^([a-zA-Z0-9_-]+):.*?## (.*)$")


def main(makefile: Path) -> None:
    for line in makefile.read_text(encoding="utf-8").splitlines():
        match = TARGET_PATTERN.match(line)
        if match:
            print(f"  {match.group(1):<22} {match.group(2)}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "Makefile"))
