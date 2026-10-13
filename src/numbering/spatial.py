"""Spatial numbering: proximity mapping, leftover matching, identical-substituent fixes."""

import copy

import torch
from rdkit.Chem.rdmolops import FragmentOnBonds

from chemistry.fragments import fragmented_to_mol
from chemistry.geometry import filter_lone
from chemistry.identities import check_identities, check_order_prochiral, correct_order
from numbering.context import RenumberingState
from optimization.optimizer import analyze_df, proximity_cutoff


def overlap_mapping(
    target_df, query_df, query_mol, dist_threshold=None,
    match_atom_type=True, occupied=None, unique_refs=False, filter_isolated=True,
):
    """Map query atoms to the nearest reference atom within `dist_threshold`.

    Default cutoff is mean |r-center| / 10, rounded to one significant digit.
    If `match_atom_type` is True, element identity is part of the distance
    (one-hot embedding). If False, only XYZ coordinates are used.
    `occupied` is an existing query→ref map whose atoms are skipped.
    """
    if dist_threshold is None:
        dist_threshold = proximity_cutoff(target_df)
    coords_t, _, full_t, atoms_dictionary = analyze_df(target_df)
    coords_q, _, full_q, _ = analyze_df(query_df, atoms_dictionary)
    target = full_t if match_atom_type else coords_t
    query = full_q if match_atom_type else coords_q

    occupied = occupied or {}
    n_query = query_mol.GetNumAtoms()
    n_ref = int(target.shape[0])
    # Only pairs whose query index exists on the RDKit mol (df can briefly
    # disagree after a bad rebuild; never invent keys past n_query).
    occupied_q = {int(q) for q in occupied if 0 <= int(q) < n_query}
    occupied_r = {int(r) for q, r in occupied.items() if 0 <= int(q) < n_query and 0 <= int(r) < n_ref}
    used_r = set(occupied_r)

    atom_mapping = {}
    for idx, point in enumerate(query):
        if idx >= n_query or idx in occupied_q:
            continue
        distances = torch.pow(torch.sum(torch.pow(target - point, 2), 1), 0.5).clone()
        if unique_refs:
            for j in used_r:
                if 0 <= int(j) < n_ref:
                    distances[int(j)] = float("inf")
        if torch.min(distances) < dist_threshold:
            j = int(torch.argmin(distances).item())
            if j >= n_ref:
                continue
            atom_mapping[idx] = j
            if unique_refs:
                used_r.add(j)
    if filter_isolated:
        return filter_lone(atom_mapping, query_mol)
    return atom_mapping


Overlap_Mapping = overlap_mapping  # backward-compatible alias


def spatial_matching_map(MCSandCoords_map, mol2, mol1, df2, df1):
    """Greedy nearest-neighbor matching of still-unmapped atoms, raising the cutoff."""
    unnumbered = {}
    mol1_atoms = len(mol1.GetAtoms())
    mol2_atoms = len(mol2.GetAtoms())
    ref_tensor = torch.tensor(df1.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
    query_tensor = torch.tensor(df2.iloc[2:, 1:].values.astype("float64")).to(torch.float64)

    for threshold in torch.arange(0.3, 5.0, 0.005):
        target_unmatched = [
            i for i in range(mol1_atoms)
            if i not in MCSandCoords_map.values() and i not in unnumbered.values()
        ]
        query_unmatched = [
            i for i in range(mol2_atoms)
            if i not in MCSandCoords_map and i not in unnumbered
        ]
        if not target_unmatched:
            break
        for i in query_unmatched:
            distances = torch.zeros(len(target_unmatched))
            for idx, j in enumerate(target_unmatched):
                distances[idx] = torch.pow(torch.sum(torch.pow(query_tensor[i] - ref_tensor[j], 2)), 0.5)
            if (
                torch.min(distances) < threshold
                and target_unmatched[torch.argmin(distances)] not in unnumbered.values()
            ):
                unnumbered[i] = target_unmatched[torch.argmin(distances)]
    return unnumbered


def spatial_corrections_map(overlap_mapping, MCSandCoords_map, mol2, df_1, df_2):
    """Reorder identical substituents (H or other equivalent groups) by local overlay."""
    update_map = {}
    mol2_atoms = mol2.GetAtoms()
    n_ref = len(df_1) - 2
    # Only pairs that point at a real reference atom (align used to invent
    # out-of-range "ref" indices when the query was larger).
    valid_map = {q: r for q, r in MCSandCoords_map.items() if 0 <= int(r) < n_ref}

    for atom_idx in valid_map:
        bonds = mol2_atoms[atom_idx].GetBonds()
        if len(bonds) < 2:
            continue
        bonded_atoms = [
            bond.GetEndAtomIdx() if bond.GetEndAtomIdx() != atom_idx else bond.GetBeginAtomIdx()
            for bond in bonds
        ]
        matched_bonded_atoms = [a for a in bonded_atoms if a in valid_map]
        if len(matched_bonded_atoms) < 2:
            continue

        mols_to_compare = []
        for r in matched_bonded_atoms:
            bonds_to_break = [b.GetIdx() for b in bonds if (b.GetEndAtomIdx() != r and b.GetBeginAtomIdx() != r)]
            mols_to_compare.append(fragmented_to_mol(FragmentOnBonds(mol2, bonds_to_break), atom_idx))

        identical_substituents = check_identities(mols_to_compare, matched_bonded_atoms)
        participating_atoms = [atom_idx] + matched_bonded_atoms
        ord_identical = [
            check_order_prochiral(s, valid_map, df_2, df_1, participating_atoms)
            if len(s) > 1 else []
            for s in identical_substituents
        ]
        for idx in range(len(ord_identical)):
            if ord_identical[idx] != list(identical_substituents[idx].keys()) and ord_identical[idx]:
                update_map = correct_order(
                    ord_identical[idx], identical_substituents[idx],
                    valid_map, update_map, overlap_mapping,
                )
    return update_map


spatial_corrections = spatial_corrections_map
spatial_matching = spatial_matching_map


def coords_proximity(state: RenumberingState) -> RenumberingState:
    """Typed proximity: map atoms that landed close after alignment (same element)."""
    cutoff = proximity_cutoff(state.df_ref)
    state.overlap_mapping = overlap_mapping(
        state.df_ref, state.df_query, state.mol_query, dist_threshold=cutoff,
        match_atom_type=True,
    )
    if not state.overlap_mapping:
        state.overlap_mapping = copy.deepcopy(
            state.atoms_mcs_map or state.mapping
        )
        state.record("coords_proximity", notes="empty proximity map; reused MCS seed")
    else:
        state.record("coords_proximity", notes=f"same-type distance < {cutoff:g} A")
    # Keep the first seed proximity map so later stages may override only these
    # soft pairs (not extend_mcs / complete_rings locks).
    if not state.seed_proximity_mapping:
        state.seed_proximity_mapping = copy.deepcopy(state.overlap_mapping)
    return state


def coords_proximity_any(state: RenumberingState) -> RenumberingState:
    """XYZ-only proximity for atoms not already set by MCS/ring stages.

    First commits the current overlap_mapping into state.mapping (typed proximity /
    ring fills win over conflicting MCS pairs). Then adds XYZ-only hits for still
    unmapped atoms. Skips query/ref atoms already in overlap_mapping and, after
    extend_mcs, atoms already in mapping. Isolated new hits are dropped.
    """
    cutoff = proximity_cutoff(state.df_ref)

    # Promote prior overlap_mapping into the live map (coords win on conflicts).
    prior = dict(state.overlap_mapping)
    old_map = dict(state.mapping)
    if prior:
        taken_q = set(prior)
        taken_r = set(prior.values())
        state.mapping = {
            q: r for q, r in state.mapping.items()
            if q not in taken_q and r not in taken_r
        }
        state.mapping.update(prior)
    promoted = sum(1 for q, r in prior.items() if old_map.get(q) != r)

    occupied = dict(state.overlap_mapping)
    ran = {entry.name for entry in state.log}
    if "extend_mcs" in ran:
        occupied.update(state.mapping)
    # complete_rings writes into overlap_mapping, so those atoms are already occupied

    raw = overlap_mapping(
        state.df_ref, state.df_query, state.mol_query, dist_threshold=cutoff,
        match_atom_type=False, occupied=occupied, unique_refs=True,
        filter_isolated=False,
    )
    combined = dict(occupied)
    combined.update(raw)
    kept = filter_lone(combined, state.mol_query)
    added = {q: r for q, r in kept.items() if q not in occupied}
    state.overlap_mapping.update(added)
    state.mapping.update(added)
    state.locked_mapping.update(added)
    note = f"+{len(added)} any-type distance < {cutoff:g} A (unmapped only; isolated dropped)"
    if promoted:
        note = f"applied {promoted} prior coords pair(s); " + note
    state.record("coords_proximity_any", notes=note)
    return state


def apply_spatial_corrections(state: RenumberingState) -> RenumberingState:
    """Numbering stage: swap identical groups/hydrogens to match reference geometry."""
    updates = spatial_corrections_map(
        state.overlap_mapping, state.mapping, state.mol_query, state.df_ref, state.df_query,
    )
    state.mapping.update(updates)
    state.record("spatial_corrections", notes=f"{len(updates)} substituent reassignments")
    return state


def apply_spatial_matching(state: RenumberingState) -> RenumberingState:
    """Numbering stage: greedy spatial match of leftover atoms."""
    added = spatial_matching_map(
        state.mapping, state.mol_query, state.mol_ref, state.df_query, state.df_ref,
    )
    state.mapping.update(added)
    state.record("spatial_matching", notes=f"+{len(added)} leftover by distance")
    return state
