"""Hydrogen numbering stage (topology: H attached to already-mapped heavy atoms)."""

from chemistry.hydrogens import extend_mapping_with_hydrogens
from numbering.context import RenumberingState


def map_hydrogens(state: RenumberingState) -> RenumberingState:
    """Numbering stage: pair hydrogens on mapped heavy atoms by neighbor order.

    Not in the default pipeline (spatial_corrections handles H geometry).
    """
    before = len(state.mapping)
    state.mapping = extend_mapping_with_hydrogens(state.mol_query, state.mol_ref, state.mapping)
    state.record("map_hydrogens", notes=f"+{len(state.mapping) - before} hydrogens")
    return state
