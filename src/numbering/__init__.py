"""Composable atom-numbering stages.

Fixed seed (MCS + align + same-element proximity), then a user-orderable
middle (`complete_rings`, `extend_mcs`, `coords_proximity_any`), then a
fixed tail from `spatial_corrections` through `permute_query`.
"""

from numbering.context import RenumberingState, StageLog
from numbering.leftover import add_unmapped, add_unumbered
from numbering.pipeline import (
    DEFAULT_MIDDLE,
    DEFAULT_ORDER,
    FIXED_END,
    FIXED_START,
    MIDDLE_ORDERS,
    MIDDLE_PERMUTATIONS,
    STAGES,
    USER_ORDERABLE,
    compose_order,
    default_middle,
    default_order,
    run_numbering,
)
from numbering.spatial import Overlap_Mapping, overlap_mapping

__all__ = [
    "Overlap_Mapping",
    "DEFAULT_MIDDLE",
    "DEFAULT_ORDER",
    "FIXED_END",
    "FIXED_START",
    "MIDDLE_ORDERS",
    "MIDDLE_PERMUTATIONS",
    "RenumberingState",
    "STAGES",
    "StageLog",
    "USER_ORDERABLE",
    "add_unmapped",
    "add_unumbered",
    "compose_order",
    "overlap_mapping",
    "default_middle",
    "default_order",
    "run_numbering",
]
