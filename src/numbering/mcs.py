"""MCS-based numbering stages."""

import copy

import torch

from chemistry.mcs import (
    apply_MCS, find_init_vars, init_pose_from_map, map_pendant_hydrogens,
    normalize_anchor_pairs,
)
from numbering.align import atoms_for_optimization, optimize_based_on_current_mapping
from numbering.context import RenumberingState
from optimization.loss import loss_function


def _alignment_loss(df_ref, df_query, atom_map, optimize_atoms=None, anchor_pairs=None):
    """Sum of mixed losses on MCS-mapped (or --fit) pairs after alignment."""
    target_atoms, query_atoms = atoms_for_optimization(
        df_ref, atom_map, optimize_atoms, anchor_pairs=anchor_pairs,
    )
    ref = torch.tensor(df_ref.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
    query = torch.tensor(df_query.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
    return float(
        torch.sum(loss_function(query[query_atoms], ref[target_atoms], sigma=50, loss_type="mixed"))
        .detach()
    )


def _mcs_map_progress(iterable, total, desc="Mapping align"):
    """tqdm progress bar when available; otherwise a simple counter on stderr."""
    try:
        from tqdm import tqdm
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit="map",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,
        )
    except ImportError:
        pass

    import sys

    class _SimpleProgress:
        def __init__(self, it, n, label):
            self._it = it
            self._n = n
            self._label = label
            self._i = 0

        def __iter__(self):
            width = max(1, len(str(self._n)))
            for item in self._it:
                self._i += 1
                pct = 100.0 * self._i / self._n if self._n else 100.0
                print(
                    f"\r{self._label}: {self._i:{width}d}/{self._n} ({pct:5.1f}%)",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
                yield item
            print(file=sys.stderr)

    return _SimpleProgress(iterable, total, desc)


def choose_init_map(
    init_vars, df_1, df_2, mol2, optimize_atoms=None, lr=None, max_iters=2000,
    anchor_pairs=None, loss_type="mixed", sigma=30, debug=False,
):
    """Try every MCS mapping; keep the one with lowest post-alignment loss.

    Also rewrites init_vars[0:3] (COM/PCA) to match the winning map.
    """
    maps = init_vars[3]
    if len(maps) == 1:
        com_q, com_r, pca = init_pose_from_map(df_1, df_2, maps[0])
        init_vars_list = list(init_vars)
        init_vars_list[0], init_vars_list[1], init_vars_list[2] = com_q, com_r, pca
        return copy.deepcopy(maps[0]), tuple(init_vars_list)

    best_idx = 0
    best_loss = float("inf")
    n_maps = len(maps)
    pbar = _mcs_map_progress(enumerate(maps), n_maps, desc="Mapping align")
    for idx, m in pbar:
        com_q, com_r, pca = init_pose_from_map(df_1, df_2, m)
        trial_init = (com_q, com_r, pca, maps, init_vars[4])
        df_2_temp = copy.deepcopy(df_2)
        temp_map = copy.deepcopy(m)
        df_2_temp, _ = optimize_based_on_current_mapping(
            df_1, df_2_temp, temp_map, mol2, trial_init,
            optimize_atoms=optimize_atoms, atoms_mcs_map=m,
            lr=lr, max_iters=max_iters, anchor_pairs=anchor_pairs,
            loss_type=loss_type, sigma=sigma, debug=debug,
        )
        loss = _alignment_loss(
            df_1, df_2_temp, m, optimize_atoms=optimize_atoms, anchor_pairs=anchor_pairs,
        )
        if loss < best_loss:
            best_loss = loss
            best_idx = idx
        # Refresh postfix when tqdm is available.
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(best=f"{best_loss:.4g}", refresh=False)

    winner = maps[best_idx]
    com_q, com_r, pca = init_pose_from_map(df_1, df_2, winner)
    init_vars_list = list(init_vars)
    init_vars_list[0], init_vars_list[1], init_vars_list[2] = com_q, com_r, pca
    return copy.deepcopy(winner), tuple(init_vars_list)


def init_mcs(state: RenumberingState, anchors=None) -> RenumberingState:
    """Numbering stage: initial MCS map plus COM/PCA seeds for alignment."""
    pairs = normalize_anchor_pairs(anchors if anchors is not None else state.anchor_pairs)
    state.anchor_pairs = pairs
    state.init_vars = find_init_vars(
        state.mol_ref, state.mol_query, state.df_ref, state.df_query, anchors=pairs,
        match_formal_charge=state.match_formal_charge,
    )
    n_maps = len(state.init_vars[3])
    state.mapping, state.init_vars = choose_init_map(
        state.init_vars, state.df_ref, state.df_query, state.mol_query,
        optimize_atoms=state.optimize_atoms,
        lr=state.lr, max_iters=state.max_iters,
        anchor_pairs=pairs,
        loss_type=getattr(state, "loss_type", "mixed"),
        sigma=getattr(state, "sigma", 30),
        debug=state.debug,
    )
    state.atoms_mcs_map = copy.deepcopy(state.mapping)
    note = f"seed map from MCS ({n_maps} candidate{'s' if n_maps != 1 else ''}; lowest loss)"
    if state.match_formal_charge:
        note += "; charge-aware"
    if pairs:
        note += "; forced " + ", ".join(f"q{q}->r{r}" for r, q in pairs)
    state.record("init_mcs", notes=note)
    return state


def _apply_locked(grown: dict, locked: dict) -> dict:
    """Keep `locked` pairs; drop any grown pair that conflicts on query or ref."""
    if not locked:
        return grown
    locked_q = set(locked)
    locked_r = set(locked.values())
    merged = {
        q: r for q, r in grown.items()
        if q not in locked_q and r not in locked_r
    }
    merged.update(locked)
    return merged


def _is_hydrogen_pair(mol_query, mol_ref, q, r) -> bool:
    return (
        mol_query.GetAtomWithIdx(int(q)).GetSymbol() == "H"
        or mol_ref.GetAtomWithIdx(int(r)).GetSymbol() == "H"
    )


def extend_mcs(state: RenumberingState) -> RenumberingState:
    """Grow the map by fragment MCS; keep complete_rings / coords_proximity_any pairs.

    When a fragment has several MCS numberings, `apply_MCS` keeps the one with
    lowest current XYZ distances (no extra rigid optimization).

    Hydrogens are never frozen here: fragment-MCS H are dropped, then refilled
    by parent-aware soft assignment so spatial_corrections can still reorder them.
    """
    locked = dict(state.locked_mapping)
    # Grow from typed proximity plus locked ring (and proximity_any) pairs so
    # complete_rings commitments are in the seed map apply_MCS iterates.
    seed = dict(state.overlap_mapping)
    seed.update(locked)
    state.overlap_mapping.update(seed)

    temp_map = apply_MCS(
        seed, state.mol_ref, state.mol_query, state.df_ref, state.df_query,
        match_formal_charge=state.match_formal_charge,
    )
    while type(temp_map) is not dict:
        temp_map = apply_MCS(
            seed, state.mol_ref, state.mol_query, state.df_ref, state.df_query,
            match_formal_charge=state.match_formal_charge,
        )
    temp_map = _apply_locked(temp_map, locked)

    # Heavy atoms only for hard/locked growth. Any H (MCS-claimed or missing)
    # goes through soft parent fill so spatial_corrections may reorder them.
    heavy = {
        q: r for q, r in temp_map.items()
        if not _is_hydrogen_pair(state.mol_query, state.mol_ref, q, r)
    }
    locked_heavy = {
        q: r for q, r in locked.items()
        if not _is_hydrogen_pair(state.mol_query, state.mol_ref, q, r)
    }
    heavy = _apply_locked(heavy, locked_heavy)
    temp_map, soft_h = map_pendant_hydrogens(
        state.mol_ref, state.mol_query, heavy, state.df_ref, state.df_query,
    )
    # Locked heavies win; locked H (if any) stay out — soft/spatial own H.
    temp_map = _apply_locked(temp_map, locked_heavy)

    for q, r in temp_map.items():
        if q in soft_h or _is_hydrogen_pair(state.mol_query, state.mol_ref, q, r):
            continue
        if seed.get(q) != r:
            state.locked_mapping[q] = r

    state.mapping = temp_map
    for q, r in temp_map.items():
        if q in soft_h or _is_hydrogen_pair(state.mol_query, state.mol_ref, q, r):
            continue
        state.overlap_mapping[q] = r
    # Ensure soft H are not frozen from an earlier coords write.
    for q in soft_h:
        state.overlap_mapping.pop(q, None)
        state.locked_mapping.pop(q, None)

    note = "fragment MCS growth (closest proximity map if several options)"
    if soft_h:
        note += f"; +{len(soft_h)} soft parent-H (reorderable)"
    if locked:
        note += f" (seed+kept {len(locked)} locked from rings/proximity_any)"
    state.record("extend_mcs", notes=note)
    return state
