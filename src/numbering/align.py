"""Rigid-body alignment of the query onto the reference using the current map."""

import torch

from chemistry.mcs import normalize_anchor_pairs
from numbering.context import RenumberingState
from optimization.optimizer import optimize_numbered, resolve_optimize_atom_pairs


def merge_fit_atoms(optimize_atoms=None, anchor_pairs=None):
    """Union of explicit --fit refs with forced-anchor ref indices.

    When `optimize_atoms` is None, returns None (fit all mapped atoms; anchors
    are already in the seed map). When `--fit` is set, anchor ref indices are
    always appended so they remain in the loss.
    """
    pairs = normalize_anchor_pairs(anchor_pairs)
    anchor_refs = [int(r) for r, _q in pairs]
    if optimize_atoms is None:
        return None
    out = list(optimize_atoms)
    for r in anchor_refs:
        if r not in out:
            out.append(r)
    return out


def atoms_for_optimization(df_target, mcs_map, optimize_atoms=None, anchor_pairs=None):
    """Reference and query atom indices to use in the alignment loss."""
    n_atoms = len(df_target) - 2
    fit_atoms = merge_fit_atoms(optimize_atoms, anchor_pairs)
    if fit_atoms is not None:
        invalid = [i for i in fit_atoms if i < 0 or i >= n_atoms]
        if invalid:
            raise ValueError(f"Reference atom indices out of range (0-{n_atoms - 1}): {invalid}")
    return resolve_optimize_atom_pairs(mcs_map, fit_atoms)


def optimize_based_on_current_mapping(
    df_1, df_2, temp_map, mol2, init_vars=None, optimize_atoms=None, atoms_mcs_map=None,
    lr=None, max_iters=2000, anchor_pairs=None, loss_type="mixed", sigma=30,
    debug=False,
):
    """Rigid-fit query onto reference using the MCS map for the loss.

    Does not invent reference indices for unmapped query atoms (that used to
    pollute `temp_map` when the query is larger than the reference).
    """
    mcs_map = atoms_mcs_map if atoms_mcs_map is not None else temp_map
    target_atoms, query_atoms = atoms_for_optimization(
        df_1, mcs_map, optimize_atoms, anchor_pairs=anchor_pairs,
    )
    df_2 = optimize_numbered(
        df_1, df_2, target_atoms, query_atoms=query_atoms, init_vars=init_vars,
        lr=lr, max_iters=max_iters, loss_type=loss_type, sigma=sigma,
        debug=debug,
    )
    return df_2, [torch.zeros(3)] * 3 + list(init_vars)[3:]


def align_to_mapping(state: RenumberingState) -> RenumberingState:
    """Numbering stage: rotate/translate query so mapped atoms overlap."""
    state.df_query, state.init_vars = optimize_based_on_current_mapping(
        state.df_ref,
        state.df_query,
        state.mapping,
        state.mol_query,
        state.init_vars,
        optimize_atoms=state.optimize_atoms,
        atoms_mcs_map=state.atoms_mcs_map,
        lr=state.lr,
        max_iters=state.max_iters,
        anchor_pairs=state.anchor_pairs,
        loss_type=getattr(state, "loss_type", "mixed"),
        sigma=getattr(state, "sigma", 30),
        debug=state.debug,
    )
    state.record("align", notes="rigid-body fit on current map")
    return state
