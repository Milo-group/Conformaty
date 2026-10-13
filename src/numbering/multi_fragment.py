"""Sequential numbering when the target has disconnected fragments."""

from __future__ import annotations

import copy
import random
import sys

import pandas as pd
from rdkit import Chem
from rdkit.Chem import GetMolFrags

from chemistry.mol import df_to_mol
from numbering.align import align_to_mapping
from numbering.context import RenumberingState
from numbering.leftover import permute_query
from numbering.pipeline import (
    FIXED_START,
    STAGES,
    default_middle,
    validate_middle,
)


# Per-fragment numbering: seed + middle + spatial cleanup, but no fill/permute.
_FRAG_TAIL = ("spatial_corrections", "spatial_matching")


def fragment_atom_groups(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """Disconnected components of `mol`, largest-first (ties broken randomly)."""
    frags = [tuple(f) for f in GetMolFrags(mol, asMols=False, sanitizeFrags=False)]
    if not frags:
        return []
    random.shuffle(frags)
    frags.sort(key=len, reverse=True)
    return frags


def df_from_atom_indices(df: pd.DataFrame, atom_indices) -> pd.DataFrame:
    """XYZ dataframe containing only `atom_indices` (in that order)."""
    atom_indices = list(atom_indices)
    cols = list(df.columns)
    rows = df.iloc[[2 + int(i) for i in atom_indices]].copy()
    rows.columns = cols
    header = pd.DataFrame([[len(atom_indices), "", "", ""]], columns=cols)
    gap = pd.DataFrame([["", "", "", ""]], columns=cols)
    out = pd.concat([header, gap, rows], ignore_index=True)
    out.columns = cols
    return out


def _run_fragment_numbering(state: RenumberingState, stage_order, anchors) -> RenumberingState:
    middle = default_middle() if stage_order is None else validate_middle(stage_order)
    full_order = FIXED_START + tuple(middle) + _FRAG_TAIL
    for name in full_order:
        stage = STAGES[name]
        if name == "seed":
            state = stage(state, anchors=anchors)
        else:
            state = stage(state)
            state.debug_dump(name)
    return state


def _translate_mapping(local_map: dict, query_globals, ref_globals) -> dict:
    """Map local query/ref indices back to the parent molecules."""
    out = {}
    n_q = len(query_globals)
    n_r = len(ref_globals)
    for lq, lr in local_map.items():
        lq, lr = int(lq), int(lr)
        if 0 <= lq < n_q and 0 <= lr < n_r:
            out[int(query_globals[lq])] = int(ref_globals[lr])
    return out


def _fill_remaining(mapping: dict, n_query: int, n_ref: int) -> dict:
    """Assign still-unmapped query atoms to unused ref indices (then extras)."""
    mapping = dict(mapping)
    used_q = set(mapping)
    used_r = set(mapping.values())
    free_r = [i for i in range(max(n_ref, n_query)) if i not in used_r]
    for q in range(n_query):
        if q in used_q:
            continue
        if not free_r:
            break
        mapping[q] = free_r.pop(0)
    return mapping


def _align_full_query_to_largest(
    full_df_ref, full_df_query, full_mol_ref, full_mol_query,
    global_mapping, lr, max_iters, match_formal_charge, debug,
) -> tuple[pd.DataFrame, object]:
    """Rigid-align the full query using only the largest-fragment mapping."""
    from chemistry.mcs import init_pose_from_map

    if not global_mapping:
        return full_df_query, None

    com_q, com_r, pca = init_pose_from_map(full_df_ref, full_df_query, global_mapping)
    init_vars = (com_q, com_r, pca, [global_mapping], None)
    state = RenumberingState(
        df_ref=full_df_ref.copy(),
        df_query=full_df_query.copy(),
        mol_ref=full_mol_ref,
        mol_query=full_mol_query,
        mapping=dict(global_mapping),
        atoms_mcs_map=dict(global_mapping),
        init_vars=init_vars,
        lr=lr,
        max_iters=max_iters,
        match_formal_charge=match_formal_charge,
        debug=debug,
    )
    state = align_to_mapping(state)
    return state.df_query, state.init_vars


def run_numbering_multi_fragment(
    state: RenumberingState, stage_order=None, anchors=None,
) -> RenumberingState:
    """Match the full query to each disconnected target fragment, largest first.

    Each target part is MCS-matched against the entire query molecule. Pairs
    that reuse a query or ref atom already assigned to an earlier part are
    dropped. If the query is ≤ the full target but larger than at least one
    part, oversized parts are skipped so smaller parts still match. Final
    query coordinates come from the largest successfully matched part.
    """
    frags = fragment_atom_groups(state.mol_ref)
    if len(frags) <= 1:
        from numbering.pipeline import run_numbering_single
        return run_numbering_single(state, stage_order=stage_order, anchors=anchors)

    sizes = [len(f) for f in frags]
    print(
        f"Multi-fragment target: {len(frags)} parts, sizes {sizes} (largest first)",
        flush=True,
    )
    state.record(
        "multi_fragment",
        notes=f"{len(frags)} parts sizes={sizes}; full-query MCS per part",
    )

    full_df_ref = state.df_ref
    full_df_query = state.df_query
    full_mol_ref = state.mol_ref
    full_mol_query = state.mol_query
    n_query = full_mol_query.GetNumAtoms()
    n_ref = full_mol_ref.GetNumAtoms()
    q_globals = list(range(n_query))

    # Query ≤ full target but larger than at least one part: skip oversized
    # parts and still match the smaller ones (instead of stopping entirely).
    match_smaller_parts = n_query <= n_ref and any(sz < n_query for sz in sizes)
    if match_smaller_parts:
        print(
            f"Query ({n_query}) ≤ target ({n_ref}) but larger than some parts; "
            "will skip oversized parts and match smaller ones",
            flush=True,
        )
        state.record(
            "multi_fragment_skip_oversized",
            notes=f"query {n_query} ≤ ref {n_ref}; match parts smaller than query",
        )

    global_mapping: dict = {}
    unmapped_query = set(range(n_query))
    final_df_query = None
    aligned_from_largest_matched = False

    for frag_i, frag_atoms in enumerate(frags):
        frag_atoms = list(frag_atoms)
        n_remaining = len(unmapped_query)
        if n_remaining == 0:
            break

        if n_remaining < len(frag_atoms) or (
            match_smaller_parts and len(frag_atoms) > n_query
        ):
            # Part too big for the query / remaining atoms.
            if match_smaller_parts:
                print(
                    f"Skip fragment {frag_i}: target {len(frag_atoms)} atoms > "
                    f"query/remaining ({n_query}/{n_remaining}); try smaller parts",
                    flush=True,
                )
                state.record(
                    f"multi_frag_{frag_i}_skip",
                    notes=(
                        f"oversized need {len(frag_atoms)}, "
                        f"query {n_query}, remaining {n_remaining}; continue"
                    ),
                )
                continue
            print(
                f"Stop at fragment {frag_i}: need {len(frag_atoms)} atoms, "
                f"only {n_remaining} query atoms left; filling leftovers",
                flush=True,
            )
            state.record(
                f"multi_frag_{frag_i}_stop",
                notes=f"need {len(frag_atoms)}, remaining {n_remaining}; fill rest",
            )
            break

        print(
            f"  fragment {frag_i}: target {len(frag_atoms)} atoms vs "
            f"full query ({n_query} atoms) ...",
            flush=True,
        )
        sys.stdout.flush()

        # Whole query each round (coords may already be aligned to the largest part).
        src_query_df = final_df_query if final_df_query is not None else full_df_query
        df_q_sub = src_query_df.copy()
        df_r_sub = df_from_atom_indices(full_df_ref, frag_atoms)
        mol_q_sub = full_mol_query
        mol_r_sub = df_to_mol(df_r_sub)

        frag_anchors = None
        if anchors:
            r_local = {g: i for i, g in enumerate(frag_atoms)}
            kept = []
            for r, q in anchors:
                if int(r) in r_local and 0 <= int(q) < n_query:
                    kept.append((r_local[int(r)], int(q)))
            frag_anchors = kept or None

        sub = RenumberingState(
            df_ref=df_r_sub,
            df_query=df_q_sub,
            mol_ref=mol_r_sub,
            mol_query=mol_q_sub,
            optimize_atoms=None,
            anchor_pairs=frag_anchors or [],
            lr=state.lr,
            max_iters=state.max_iters,
            match_formal_charge=state.match_formal_charge,
            debug=state.debug,
        )
        sub = _run_fragment_numbering(sub, stage_order, frag_anchors)

        # Drop any pair that reuses an atom already taken by an earlier part.
        used_q = set(global_mapping)
        used_r = set(global_mapping.values())
        local_map = {}
        dropped = 0
        for lq, lr in sub.mapping.items():
            lq, lr = int(lq), int(lr)
            if not (0 <= lq < n_query and 0 <= lr < len(frag_atoms)):
                dropped += 1
                continue
            gq = q_globals[lq]
            gr = frag_atoms[lr]
            if gq in used_q or gr in used_r:
                dropped += 1
                continue
            local_map[lq] = lr
            used_q.add(gq)
            used_r.add(gr)

        translated = _translate_mapping(local_map, q_globals, frag_atoms)
        global_mapping.update(translated)
        unmapped_query -= set(translated)

        state.record(
            f"multi_frag_{frag_i}",
            notes=(
                f"size {len(frag_atoms)}; +{len(translated)} pairs "
                f"(dropped {dropped} already-mapped; query left {len(unmapped_query)})"
            ),
        )
        print(
            f"  fragment {frag_i}: +{len(translated)} mapped "
            f"(dropped {dropped} already-mapped), "
            f"{len(unmapped_query)} query left",
            flush=True,
        )

        # Final coords from the largest successfully matched part (first hit
        # in largest-first order, including after skipped oversized parts).
        if not aligned_from_largest_matched and translated:
            aligned_from_largest_matched = True
            final_df_query, init_vars = _align_full_query_to_largest(
                full_df_ref,
                full_df_query,
                full_mol_ref,
                full_mol_query,
                dict(translated),
                state.lr,
                state.max_iters,
                state.match_formal_charge,
                state.debug,
            )
            state.init_vars = init_vars
            state.atoms_mcs_map = copy.deepcopy(translated)

    if final_df_query is None:
        final_df_query = full_df_query

    state.df_ref = full_df_ref
    state.mol_ref = full_mol_ref
    state.df_query = final_df_query
    state.mol_query = full_mol_query
    state.mapping = global_mapping
    state.overlap_mapping = dict(global_mapping)

    before = len(state.mapping)
    state.mapping = _fill_remaining(state.mapping, n_query, n_ref)
    state.record(
        "fill_unmapped",
        notes=f"+{len(state.mapping) - before} sequential (multi-frag leftovers)",
    )
    state.debug_dump("fill_unmapped")
    state = permute_query(state)
    state.debug_dump("permute_query")
    return state
