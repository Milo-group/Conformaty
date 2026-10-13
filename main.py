#!/usr/bin/env python
"""Run Conformerity from this folder.

    python main.py align -t test_cases/01_structural_alignment/cyclohexane_chair.xyz -q test_cases/01_structural_alignment/cyclohexane_boat.xyz
    python main.py pair -t test_cases/03_ring_numbering/stage_order_ref.xyz -q test_cases/03_ring_numbering/stage_order_query.xyz
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

from cli import main as run_cli


def run(argv: list[str] | None = None) -> int:
    return run_cli(argv, base_output=str(OUTPUTS))


if __name__ == "__main__":
    sys.exit(run())
