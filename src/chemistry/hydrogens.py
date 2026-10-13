"""Hydrogen-atom mapping helpers."""


def extend_mapping_with_hydrogens(query_mol, ref_mol, heavy_atom_mapping):
    """Map hydrogens attached to already-mapped heavy atoms, in neighbor order."""
    extended_mapping = dict(heavy_atom_mapping)
    for query_idx, ref_idx in heavy_atom_mapping.items():
        query_atom = query_mol.GetAtomWithIdx(query_idx)
        ref_atom = ref_mol.GetAtomWithIdx(ref_idx)
        query_h = [nbr.GetIdx() for nbr in query_atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
        ref_h = [nbr.GetIdx() for nbr in ref_atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
        for qh_idx, rh_idx in zip(query_h, ref_h):
            extended_mapping[qh_idx] = rh_idx
    return extended_mapping
