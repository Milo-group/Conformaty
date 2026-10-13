"""Numbering pipeline: fixed seed, user-orderable middle, fixed tail.

Seed (always first): MCS + rigid align + same-element proximity.
Middle: any non-empty permutation of complete_rings, extend_mcs,
coords_proximity_any.
Tail (always last): spatial_corrections onward.
"""

import itertools

from numbering.align import align_to_mapping
from numbering.context import RenumberingState
from numbering.hydrogens import map_hydrogens
from numbering.leftover import fill_unmapped, permute_query
from numbering.mcs import extend_mcs, init_mcs
from numbering.rings import complete_rings
from numbering.spatial import (
    apply_spatial_corrections,
    apply_spatial_matching,
    coords_proximity,
    coords_proximity_any,
)


def seed(state: RenumberingState, anchors=None) -> RenumberingState:
    """Fixed start: MCS seed, rigid-body fit, same-element proximity."""
    state = init_mcs(state, anchors=anchors)
    state.debug_dump("init_mcs")
    state = align_to_mapping(state)
    state.debug_dump("align")
    state = coords_proximity(state)
    state.debug_dump("coords_proximity")
    return state


STAGES = {
    "seed": seed,
    "complete_rings": complete_rings,
    "extend_mcs": extend_mcs,
    "coords_proximity_any": coords_proximity_any,
    "spatial_corrections": apply_spatial_corrections,
    "map_hydrogens": map_hydrogens,
    "spatial_matching": apply_spatial_matching,
    "fill_unmapped": fill_unmapped,
    "permute_query": permute_query,
}

FIXED_START = ("seed",)

USER_ORDERABLE = (
    "complete_rings",
    "extend_mcs",
    "coords_proximity_any",
)

FIXED_END = (
    "spatial_corrections",
    "spatial_matching",
    "fill_unmapped",
    "permute_query",
)

# Default middle: rings, then MCS growth, then any-type proximity.
DEFAULT_MIDDLE = USER_ORDERABLE

# All 6 orderings of the three middle stages.
MIDDLE_PERMUTATIONS = tuple(itertools.permutations(USER_ORDERABLE))

# All non-empty subset orderings (1-, 2-, and 3-stage).
MIDDLE_ORDERS = tuple(
    perm
    for k in range(1, len(USER_ORDERABLE) + 1)
    for combo in itertools.combinations(USER_ORDERABLE, k)
    for perm in itertools.permutations(combo)
)


def default_middle() -> tuple:
    return DEFAULT_MIDDLE


def compose_order(middle=None) -> tuple:
    """Full pipeline: seed + user middle + fixed tail."""
    if middle is None:
        middle = default_middle()
    return FIXED_START + tuple(middle) + FIXED_END


def validate_middle(stage_order) -> tuple:
    """Accept any unique subset/permutation of USER_ORDERABLE."""
    middle = tuple(stage_order)
    if not middle:
        raise ValueError(
            f"--stages must list at least one of: {list(USER_ORDERABLE)}"
        )
    seen = set()
    for name in middle:
        if name not in USER_ORDERABLE:
            raise ValueError(
                f"Unknown or non-orderable stage {name!r}. "
                f"Choosable: {list(USER_ORDERABLE)}"
            )
        if name in seen:
            raise ValueError(f"Duplicate stage {name!r}")
        seen.add(name)
    return middle


def default_order() -> tuple:
    """Full default pipeline (seed + middle + tail)."""
    return compose_order()


DEFAULT_ORDER = compose_order()


def run_numbering_single(
    state: RenumberingState, stage_order=None, anchors=None,
) -> RenumberingState:
    """Run the single-target numbering pipeline (no multi-fragment split)."""
    middle = default_middle() if stage_order is None else validate_middle(stage_order)
    full_order = compose_order(middle)
    for name in full_order:
        stage = STAGES[name]
        if name == "seed":
            state = stage(state, anchors=anchors)
        else:
            state = stage(state)
            state.debug_dump(name)
    return state


def run_numbering(state: RenumberingState, stage_order=None, anchors=None) -> RenumberingState:
    """Run numbering. `stage_order` is the user middle only (not seed/tail).

    If the target has multiple disconnected fragments, match them sequentially
    (largest first); otherwise run the usual single-structure pipeline.
    """
    from rdkit.Chem import GetMolFrags

    from numbering.multi_fragment import run_numbering_multi_fragment

    n_frags = len(GetMolFrags(state.mol_ref, asMols=False, sanitizeFrags=False))
    if n_frags > 1:
        return run_numbering_multi_fragment(
            state, stage_order=stage_order, anchors=anchors,
        )
    return run_numbering_single(state, stage_order=stage_order, anchors=anchors)
