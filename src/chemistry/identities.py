"""Identical-substituent detection and geometry-based map selection."""

import itertools

import pandas as pd
import torch

from optimization.optimizer import optimize_numbered


def check_identities(mols, matched_bonded_atoms):
    unique_structures = len(mols)
    identity_list = []

    for i in range(len(mols)):
        if matched_bonded_atoms[i] in [d for n in identity_list for d in n]:
            continue
        identity_list.append({matched_bonded_atoms[i]: mols[i][1]})
        temp_ids = 1
        for j in range(i + 1, len(mols)):
            if (
                mols[i][0].HasSubstructMatch(mols[j][0], useChirality=True)
                and mols[j][0].HasSubstructMatch(mols[i][0], useChirality=True)
            ):
                temp_ids += 1
                identity_list[-1][matched_bonded_atoms[j]] = {
                    idx: mols[j][1][atom]
                    for (idx, atom) in enumerate(mols[j][0].GetSubstructMatch(mols[i][0]))
                }

        unique_structures -= temp_ids
        if unique_structures < 2:
            break

    return identity_list


def check_order(s, new_map, df2, df1):
    """Deprecated: std of pairwise distances. Prefer `check_order_prochiral`."""
    s = s.keys()
    ref_tensor = torch.tensor(df1.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
    ref_s = [new_map[i] for i in s]
    query_tensor = torch.tensor(df2.iloc[2:, 1:].values.astype("float64")).to(torch.float64)

    permutations = [list(i) for i in itertools.permutations(s)]
    distances = torch.zeros(len(permutations))
    for idx, perm in enumerate(permutations):
        distances[idx] = torch.std(
            torch.pow(torch.sum(torch.pow(query_tensor[perm] - ref_tensor[ref_s], 2), 1), 0.5)
        )
    return permutations[torch.argmin(distances)]


def check_order_prochiral(s, new_map, df2, df1, participating_atoms):
    """Choose the permutation of identical groups (H or otherwise) that overlays best.

    For each permutation, rigid-fits the local fragment (center + neighbors) onto the
    reference and scores by sum of residual distances, not std of pairwise distances.
    """
    s_keys = list(s.keys())
    if any(k not in new_map for k in s_keys):
        return s_keys
    n_ref = len(df1) - 2
    n_query = len(df2) - 2
    permutations = [list(p) for p in itertools.permutations(s_keys)]
    distances = torch.zeros(len(permutations))

    non_identical_atoms = [
        i for i in participating_atoms
        if i not in s and i in new_map and 0 <= int(i) < n_query and 0 <= int(new_map[i]) < n_ref
    ]
    non_identical_atoms_ref = [new_map[i] for i in non_identical_atoms]
    ref_s = [new_map[i] for i in s_keys]
    if any(r < 0 or r >= n_ref for r in ref_s):
        return s_keys

    for idx, perm in enumerate(permutations):
        n_atoms = len(non_identical_atoms) + len(perm)
        df_1_temp = pd.concat([
            df1.iloc[:2, :],
            df1.iloc[2:, :].iloc[non_identical_atoms_ref + ref_s, :],
        ])
        df_2_temp = pd.concat([
            df1.iloc[:2, :],
            df2.iloc[2:, :].iloc[non_identical_atoms + perm, :],
        ])
        df_1_temp.index = [0, 0] + list(range(n_atoms))
        df_2_temp.index = [0, 0] + list(range(n_atoms))

        df_2_temp = optimize_numbered(
            df_1_temp, df_2_temp, list(range(n_atoms)), init_vars=None
        )

        ref_tensor = torch.tensor(df_1_temp.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
        query_tensor = torch.tensor(df_2_temp.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
        distances[idx] = torch.sum(
            torch.pow(torch.sum(torch.pow(query_tensor - ref_tensor, 2), 1), 0.5)
        )

    return permutations[int(torch.argmin(distances))]


def choose_map(MCS_maps, frag_ref_mapping, frag_query_mapping, df2, df1):
    """Pick the fragment MCS map closest in XYZ proximity (no optimization).

    Primary key: mean pairwise Euclidean distance (lower is better).
    Tie-break: more mapped atoms.
    """
    if not MCS_maps:
        return {}
    if len(MCS_maps) == 1:
        return MCS_maps[0]

    ref_tensor = torch.tensor(df1.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
    query_tensor = torch.tensor(df2.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
    n_ref = int(ref_tensor.shape[0])
    n_query = int(query_tensor.shape[0])

    best = MCS_maps[0]
    best_key = None  # (mean_distance, -n_atoms)

    for mapping in MCS_maps:
        if not mapping:
            continue
        pairs = []
        valid = True
        for i, j in mapping.items():
            q = frag_query_mapping[i]
            r = frag_ref_mapping[j]
            if not (0 <= q < n_query and 0 <= r < n_ref):
                valid = False
                break
            pairs.append((q, r))
        if not valid or not pairs:
            continue
        q_idx = [p[0] for p in pairs]
        r_idx = [p[1] for p in pairs]
        dist_sum = float(
            torch.sum(
                torch.pow(torch.sum(torch.pow(query_tensor[q_idx] - ref_tensor[r_idx], 2), 1), 0.5)
            ).detach()
        )
        mean_dist = dist_sum / len(pairs)
        key = (mean_dist, -len(pairs))
        if best_key is None or key < best_key:
            best_key = key
            best = mapping

    return best


def compare_to_previous(MCS_map, MCSandCoords_map, overlap_mapping, frag_query_mapping, frag_ref_mapping, df2, df1):
    common_new_frag = [
        n for n in MCS_map
        if frag_ref_mapping[MCS_map[n]] in MCSandCoords_map.values()
        and frag_ref_mapping[MCS_map[n]] not in overlap_mapping.values()
    ]
    common_new = [frag_query_mapping[n] for n in common_new_frag]
    common_previous = [
        n for n in MCSandCoords_map
        if MCSandCoords_map[n] in [frag_ref_mapping[MCS_map[i]] for i in MCS_map]
        and MCSandCoords_map[n] not in overlap_mapping.values()
    ]
    common_ref = [
        MCSandCoords_map[n] for n in MCSandCoords_map
        if MCSandCoords_map[n] in [frag_ref_mapping[MCS_map[i]] for i in MCS_map]
        and MCSandCoords_map[n] not in overlap_mapping.values()
    ]

    ref_tensor = torch.tensor(df1.iloc[2:, 1:].values.astype("float64")).to(torch.float64)
    query_tensor = torch.tensor(df2.iloc[2:, 1:].values.astype("float64")).to(torch.float64)

    dist_new = torch.sum(torch.pow(torch.sum(torch.pow(query_tensor[common_new] - ref_tensor[common_ref], 2), 1), 0.5))
    dist_previous = torch.sum(
        torch.pow(torch.sum(torch.pow(query_tensor[common_previous] - ref_tensor[common_ref], 2), 1), 0.5)
    )

    if dist_new > dist_previous:
        for i in common_new_frag:
            del MCS_map[i]

    return MCS_map


def correct_order(new_order, existing_order, MCSandCoords_map, update_map, overlap_mapping):
    for idx, n in enumerate(new_order):
        init_dic = existing_order[list(existing_order.keys())[idx]]
        target_dic = existing_order[n]

        if init_dic == target_dic or len(init_dic) != len(target_dic):
            continue

        changes = [init_dic[i] for i in init_dic if init_dic[i] != target_dic[i]]
        changes += [target_dic[i] for i in target_dic if init_dic[i] != target_dic[i]]

        if len(changes & overlap_mapping.keys()) > 1:
            continue

        for i in init_dic:
            if init_dic[i] in update_map:
                continue
            if init_dic[i] == target_dic[i]:
                continue
            if init_dic[i] in MCSandCoords_map and target_dic[i] in MCSandCoords_map:
                update_map[init_dic[i]] = MCSandCoords_map[target_dic[i]]

    return update_map
