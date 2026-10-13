"""Shared state passed between numbering stages.

Each stage reads/writes this object. Seed is always first; spatial cleanup
always last. The user middle (`complete_rings`, `extend_mcs`,
`coords_proximity_any`) can be reordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from rdkit import Chem


@dataclass
class StageLog:
    """One numbering-stage snapshot."""

    name: str
    mapped: int
    total: int
    notes: str = ""

    def summary(self) -> str:
        extra = f"  {self.notes}" if self.notes else ""
        return f"{self.name:24s}  mapped {self.mapped}/{self.total}{extra}"


@dataclass
class RenumberingState:
    """Query/reference pair plus the evolving query->ref atom map."""

    df_ref: pd.DataFrame
    df_query: pd.DataFrame
    mol_ref: Chem.Mol
    mol_query: Chem.Mol
    mapping: dict = field(default_factory=dict)
    overlap_mapping: dict = field(default_factory=dict)
    # Snapshot of the first (seed) typed coords_proximity map; overridable by rings.
    seed_proximity_mapping: dict = field(default_factory=dict)
    # Pairs from complete_rings / extend_mcs / coords_proximity_any — not overridable.
    locked_mapping: dict = field(default_factory=dict)
    init_vars: Any = None
    atoms_mcs_map: Optional[dict] = None
    # Reference atom indices used in alignment loss (--fit / --anchors).
    optimize_atoms: Optional[list] = None
    # Forced (ref_idx, query_idx) pairs injected into the seed map.
    anchor_pairs: list = field(default_factory=list)
    lr: Optional[float] = None
    max_iters: int = 2000
    # Optional MCS formal-charge / radical matching (off by default).
    match_formal_charge: bool = False
    debug: bool = False
    log: list = field(default_factory=list)

    @property
    def n_query(self) -> int:
        return len(self.mol_query.GetAtoms()) if self.mol_query else 0

    def record(self, name: str, notes: str = "") -> None:
        self.log.append(StageLog(name, len(self.mapping), self.n_query, notes))

    def _format_map(self, mapping: dict) -> str:
        if not mapping:
            return "    (empty)"
        lines = []
        for q, r in sorted(mapping.items()):
            q_sym = self.mol_query.GetAtomWithIdx(int(q)).GetSymbol() if self.mol_query else "?"
            r_sym = "?"
            if self.mol_ref is not None and 0 <= int(r) < self.mol_ref.GetNumAtoms():
                r_sym = self.mol_ref.GetAtomWithIdx(int(r)).GetSymbol()
            lines.append(f"    q{int(q)}({q_sym}) -> r{int(r)}({r_sym})")
        return "\n".join(lines)

    def debug_dump(self, stage_name: str) -> None:
        if not self.debug:
            return
        print(f"\n[debug] after {stage_name}  mapping {len(self.mapping)}/{self.n_query}")
        print("  mapping (query -> ref):")
        print(self._format_map(self.mapping))
        print(f"  overlap_mapping ({len(self.overlap_mapping)}):")
        print(self._format_map(self.overlap_mapping))
        print(flush=True)

    def workflow_text(self) -> str:
        lines = [
            "Renumbering workflow (query index -> reference index)",
            f"Query atoms: {self.n_query}",
            "",
        ]
        lines.extend(entry.summary() for entry in self.log)
        return "\n".join(lines) + "\n"
