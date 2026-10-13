"""Fill leftover atoms and apply the final permutation to the query XYZ."""

from numbering.context import RenumberingState


def add_unmapped(mapping, mol_query):
    """Assign sequential unused reference indices to still-unmapped query atoms."""
    mol2_atoms = mol_query.GetAtoms()
    unnumbered = {}
    for idx2 in range(len(mol2_atoms)):
        if idx2 not in mapping:
            for idx in range(len(mol2_atoms)):
                if idx not in mapping.values() and idx not in unnumbered.values():
                    unnumbered[idx2] = idx
                    break
    return unnumbered


add_unumbered = add_unmapped


def fill_unmapped(state: RenumberingState) -> RenumberingState:
    """Numbering stage: sequential leftovers so every query atom has an index."""
    added = add_unmapped(state.mapping, state.mol_query)
    state.mapping.update(added)
    state.record("fill_unmapped", notes=f"+{len(added)} sequential")
    return state


def permute_query(state: RenumberingState) -> RenumberingState:
    """Reorder query dataframe rows by ascending reference index.

    Works for a dense 0..n-1 permutation and for sparse/high ref indices
    (e.g. multi-fragment targets larger than the query).
    """
    order = [
        q for q, _r in sorted(state.mapping.items(), key=lambda qr: (qr[1], qr[0]))
    ]
    state.df_query.iloc[2:, :] = state.df_query.iloc[2:, :].iloc[order, :]
    state.record("permute_query", notes="query rows reordered")
    return state
