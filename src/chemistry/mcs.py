"""Maximum common substructure (MCS) mapping."""

import copy

import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmarts, rdFMCS
from rdkit.Chem.rdmolops import FragmentOnBonds

from chemistry.fmcs import fmcs
from chemistry.fragments import even_fragments, fragmented_to_mol
from chemistry.geometry import find_best_fitted_surface, rotation_angles_between_vectors
from chemistry.identities import choose_map, compare_to_previous


def _connected_from_seed(mol, mapped_idxs, seed):
    """Mapped query atoms reachable from `seed` through mapped neighbors only."""
    mapped = set(int(i) for i in mapped_idxs)
    seed = int(seed)
    if seed not in mapped:
        return set()
    keep = {seed}
    stack = [seed]
    while stack:
        i = stack.pop()
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            if j in mapped and j not in keep:
                keep.add(j)
                stack.append(j)
    return keep


def produce_maps_list(
    ref,
    query,
    main_idx,
    ref_main_idx,
    bondCommpare=Chem.rdFMCS.BondCompare.CompareOrder,
    atomCompare=Chem.rdFMCS.AtomCompare.CompareElements,
    match_formal_charge=False,
):
    """MCS maps between two fragments, always including the spatially matched seed.

    Full-fragment matches are preferred when they exist. Otherwise a partial MCS
    is kept if it contains `main_idx -> ref_main_idx` and is a connected
    component of that seed (so a Br vs H substituent does not discard the ring).
    """
    atom_typer = "elements" if match_formal_charge else "elements_only"
    mcs_out = fmcs([ref, query], maximize="bonds_multiple", atomCompare=atom_typer)
    smarts = getattr(mcs_out, "smarts", None) if mcs_out is not None else None
    if not smarts:
        return [{}]

    maps_list = []
    for mcs_smarts in smarts.split("..."):
        mcs_mol = MolFromSmarts(mcs_smarts)
        if mcs_mol is None:
            continue
        n_mcs = mcs_mol.GetNumAtoms()
        # uniquify=False: keep alternate ring/chain numberings for distance selection
        ref_matches = list(ref.GetSubstructMatches(mcs_mol, uniquify=False, maxMatches=256))
        query_matches = list(query.GetSubstructMatches(mcs_mol, uniquify=False, maxMatches=256))

        for q_map in query_matches:
            for r_map in ref_matches:
                if len(q_map) != n_mcs or len(r_map) != n_mcs:
                    continue
                mapping = {q_map[i]: r_map[i] for i in range(n_mcs)}
                if main_idx not in mapping or mapping[main_idx] != ref_main_idx:
                    continue
                keep = _connected_from_seed(query, mapping, main_idx)
                mapping = {q: mapping[q] for q in keep}
                if mapping and mapping not in maps_list:
                    maps_list.append(mapping)

    return maps_list if maps_list else [{}]


def _heavy_atom_index_map(mol):
    """Map RemoveHs atom index -> original mol atom index (non-H only)."""
    org = {}
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() != "H":
            org[len(org)] = i
    return org


def atom_imbalance(atom):
    """Formal charge, else radical count — treated as the same imbalance label."""
    fc = int(atom.GetFormalCharge())
    if fc != 0:
        return fc
    rad = int(atom.GetNumRadicalElectrons())
    return rad if rad else 0


def mol_for_mcs(mol):
    """Copy with radical electrons folded into formal charge for MCS matching."""
    out = Chem.RWMol(mol)
    for atom in out.GetAtoms():
        imb = atom_imbalance(atom)
        atom.SetFormalCharge(imb)
        atom.SetNumRadicalElectrons(0)
    return out.GetMol()


def _full_to_match_index(mol, with_hs):
    """Map full-molecule atom index -> index in the MCS match mol."""
    if with_hs:
        return {i: i for i in range(mol.GetNumAtoms())}
    out = {}
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() != "H":
            out[i] = len(out)
    return out


def equalize_anchors_for_mcs(ref_match, query_match, ref, query, anchor_pairs, with_hs=False):
    """Make each anchor pair identical (element + charge/radical) on MCS copies.

    Query atoms are retagged to match their reference partners so FindMCS /
    MatchFormalCharge can include the pair. Original mols are unchanged.
    """
    pairs = normalize_anchor_pairs(anchor_pairs)
    if not pairs:
        return ref_match, query_match

    ref_idx = _full_to_match_index(ref, with_hs)
    query_idx = _full_to_match_index(query, with_hs)
    ref_rw = Chem.RWMol(ref_match)
    query_rw = Chem.RWMol(query_match)

    for ref_a, query_a in pairs:
        if ref_a not in ref_idx or query_a not in query_idx:
            print(
                f"MCS equalize skip q{query_a}->r{ref_a} "
                "(not present on heavy-atom MCS mol)"
            )
            continue
        ra = ref_rw.GetAtomWithIdx(ref_idx[ref_a])
        qa = query_rw.GetAtomWithIdx(query_idx[query_a])
        old = (qa.GetSymbol(), int(qa.GetFormalCharge()), int(qa.GetNumRadicalElectrons()))
        qa.SetAtomicNum(int(ra.GetAtomicNum()))
        qa.SetFormalCharge(int(ra.GetFormalCharge()))
        qa.SetNumRadicalElectrons(int(ra.GetNumRadicalElectrons()))
        new = (qa.GetSymbol(), int(qa.GetFormalCharge()), int(qa.GetNumRadicalElectrons()))
        if old != new:
            print(
                f"MCS equalize q{query_a}->r{ref_a}: "
                f"query {old[0]}/fc={old[1]} -> {new[0]}/fc={new[1]} "
                f"(matched to ref for MCS only)"
            )
    return ref_rw.GetMol(), query_rw.GetMol()


def normalize_anchor_pairs(anchors):
    """Normalize anchors to a list of (ref_idx, query_idx) pairs (0-based).

    Accepted forms:
      - None / []
      - (ref, query) or [ref, query]
      - [(ref, query), ...]
      - {"ref": [...], "query": [...]}  (parallel lists)
      - {query_idx: ref_idx, ...}       (query->ref dict)
    """
    if anchors is None:
        return []
    if isinstance(anchors, dict):
        if "ref" in anchors or "query" in anchors:
            refs = list(anchors.get("ref") or [])
            queries = list(anchors.get("query") or [])
            if len(refs) != len(queries):
                raise ValueError(
                    f"anchors ref/query length mismatch: {len(refs)} vs {len(queries)}"
                )
            return [(int(r), int(q)) for r, q in zip(refs, queries)]
        return [(int(r), int(q)) for q, r in anchors.items()]
    if isinstance(anchors, (list, tuple)):
        if len(anchors) == 0:
            return []
        if len(anchors) == 2 and not isinstance(anchors[0], (list, tuple, dict)):
            return [(int(anchors[0]), int(anchors[1]))]
        pairs = []
        for item in anchors:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    f"Each anchor pair must be (ref_idx, query_idx); got {item!r}"
                )
            pairs.append((int(item[0]), int(item[1])))
        return pairs
    raise ValueError(f"Unsupported anchors format: {type(anchors)!r}")


def force_anchor(mapping, anchors, n_ref, n_query):
    """Force one query->ref pair into a map; drop conflicting MCS pairs.

    `anchors` is (ref_idx, query_idx), 0-based full-molecule indices.
    Works even when the pair is absent from MCS (e.g. different elements).
    """
    ref_a, query_a = int(anchors[0]), int(anchors[1])
    if not (0 <= ref_a < n_ref and 0 <= query_a < n_query):
        raise ValueError(
            f"Anchor indices out of range: ref={ref_a} (n={n_ref}), "
            f"query={query_a} (n={n_query})"
        )
    out = {q: r for q, r in mapping.items() if q != query_a and r != ref_a}
    out[query_a] = ref_a
    return out


def force_anchors(mapping, anchors, n_ref, n_query):
    """Force all anchor pairs into a map (see `normalize_anchor_pairs`)."""
    out = dict(mapping)
    for pair in normalize_anchor_pairs(anchors):
        out = force_anchor(out, pair, n_ref, n_query)
    return out


def find_init_vars(
    ref, query, df_1, df_2, Frag_mcs=-1, anchors=None, with_hs=False,
    match_formal_charge=False,
):
    """Return (COM_query, COM_ref, PCA_angles, mapping_list, ring_info).

    By default MCS search and mapping enumeration ignore hydrogens
    (`with_hs=False`). Indices in the returned maps are still full-molecule
    (with-H) atom indices.

    If `match_formal_charge` is True (optional; off by default), charge/radical
    imbalance is treated as one label: MCS copies fold radicals into formal
    charge and FindMCS uses MatchFormalCharge (useful for nature-like C⁺ cases).

    If `anchors` is set (one or more (ref, query) pairs), those pairs are always
    injected into every seed map. When charge matching is on, each pair is also
    equalized for the MCS search only.
    """
    anchor_pairs = normalize_anchor_pairs(anchors)
    if with_hs:
        base_query, base_ref = query, ref
        ref_org_map = {i: i for i in range(ref.GetNumAtoms())}
        query_org_map = {i: i for i in range(query.GetNumAtoms())}
    else:
        base_query = Chem.rdmolops.RemoveHs(query)
        base_ref = Chem.rdmolops.RemoveHs(ref)
        ref_org_map = _heavy_atom_index_map(ref)
        query_org_map = _heavy_atom_index_map(query)

    if match_formal_charge:
        query_to_match = mol_for_mcs(base_query)
        ref_to_match = mol_for_mcs(base_ref)
    else:
        query_to_match = Chem.Mol(base_query)
        ref_to_match = Chem.Mol(base_ref)

    if anchor_pairs and match_formal_charge:
        ref_to_match, query_to_match = equalize_anchors_for_mcs(
            ref_to_match, query_to_match, ref, query, anchor_pairs, with_hs=with_hs,
        )

    if match_formal_charge:
        n_imb = sum(1 for a in ref_to_match.GetAtoms() if a.GetFormalCharge()) + sum(
            1 for a in query_to_match.GetAtoms() if a.GetFormalCharge()
        )
        if n_imb:
            print(f"MCS charge/radical tags: {n_imb} imbalanced heavy atoms")

    print("search for MCS...")
    params = rdFMCS.MCSParameters()
    params.AtomCompareParameters.MatchFormalCharge = bool(match_formal_charge)
    params.BondTyper = rdFMCS.BondCompare.CompareAny
    params.Timeout = 120
    mcs = rdFMCS.FindMCS([ref_to_match, query_to_match], params).smartsString
    if not mcs:
        if not anchor_pairs:
            raise ValueError(
                "No MCS found"
                + (" (charge/radical-aware)" if match_formal_charge else "")
            )
        print("No MCS found; seeding from anchors only")
        mcs_mol = None
        mapping = [{}]
    else:
        mcs_mol = Chem.MolFromSmarts(mcs)
        print(f"mcs is {Chem.MolToSmiles(mcs_mol)}")

        if Frag_mcs >= 0:
            mcs_mol = even_fragments(mcs_mol)[Frag_mcs]

        ref_mcs_maps = list(ref_to_match.GetSubstructMatches(mcs_mol, uniquify=False))
        query_mcs_maps = list(query_to_match.GetSubstructMatches(mcs_mol, uniquify=False))
        if not ref_mcs_maps or not query_mcs_maps:
            if not anchor_pairs:
                raise ValueError("MCS smarts did not match both molecules")
            print("MCS smarts unmatched; seeding from anchors only")
            mapping = [{}]
            mcs_mol = None
        else:
            mapping = []
            for r_map in ref_mcs_maps:
                for q_map in query_mcs_maps:
                    # When charge matching is on, keep only maps where imbalance
                    # labels agree (safety if SMARTS omits charge).
                    if match_formal_charge and any(
                        atom_imbalance(query_to_match.GetAtomWithIdx(q_map[i]))
                        != atom_imbalance(ref_to_match.GetAtomWithIdx(r_map[i]))
                        for i in range(len(q_map))
                    ):
                        continue
                    upt_query_map = [query_org_map[i] for i in q_map]
                    upt_ref_map = [ref_org_map[i] for i in r_map]
                    m = {
                        upt_query_map[i]: upt_ref_map[i]
                        for i in range(len(mcs_mol.GetAtoms()))
                    }
                    if all(n != m for n in mapping):
                        mapping.append(m)

            if not mapping:
                if not anchor_pairs:
                    raise ValueError("No MCS mappings found")
                print("No MCS mappings after filters; seeding from anchors only")
                mapping = [{}]
                mcs_mol = None

    # Ring lists use MCS atom order before anchors are injected.
    ring_info = []
    if mcs_mol is not None and mapping and mapping[0]:
        upt_query_map = list(mapping[0].keys())
        if len(upt_query_map) >= mcs_mol.GetNumAtoms():
            ring_info = [
                [upt_query_map[i] for i in ring] for ring in Chem.GetSSSR(mcs_mol)
            ]

    if anchor_pairs:
        n_ref = ref.GetNumAtoms()
        n_query = query.GetNumAtoms()
        forced = []
        outside = 0
        for m in mapping:
            missing = [
                (r, q) for r, q in anchor_pairs
                if q not in m or m[q] != r
            ]
            fm = force_anchors(m, anchor_pairs, n_ref, n_query)
            if missing:
                outside += 1
            if fm not in forced:
                forced.append(fm)
        mapping = forced
        bits = []
        for ref_a, query_a in anchor_pairs:
            r_sym = ref.GetAtomWithIdx(ref_a).GetSymbol()
            q_sym = query.GetAtomWithIdx(query_a).GetSymbol()
            bits.append(f"q{query_a}({q_sym})->r{ref_a}({r_sym})")
        note = "anchors forced " + ", ".join(bits)
        if outside:
            note += f" ({outside} map(s) had anchor(s) outside / mismatched MCS)"
        print(note)

    print(f"MCS mappings: {len(mapping)} (heavy atoms only)" if not with_hs else f"MCS mappings: {len(mapping)}")

    COM_2, COM_1, PCA_angles = init_pose_from_map(df_1, df_2, mapping[0])
    return COM_2, COM_1, PCA_angles, mapping, ring_info


def init_pose_from_map(df_1, df_2, atom_map):
    """COM/PCA seed for rigid alignment from one query->ref map."""
    query_idxs = list(atom_map.keys())
    ref_idxs = [atom_map[q] for q in query_idxs]
    df_2_tensor = torch.tensor(df_2.iloc[2:, 1:].values.astype("float64"))
    df_1_tensor = torch.tensor(df_1.iloc[2:, 1:].values.astype("float64"))
    COM_2 = torch.sum(df_2_tensor[query_idxs], 0) / len(query_idxs)
    COM_1 = torch.sum(df_1_tensor[ref_idxs], 0) / len(ref_idxs)
    PCA2 = find_best_fitted_surface(df_2_tensor[query_idxs])
    PCA1 = find_best_fitted_surface(df_1_tensor[ref_idxs])
    PCA_angles = rotation_angles_between_vectors(PCA1, PCA2)
    return COM_2, COM_1, PCA_angles


def bonds_to_cut_for_extension(mol, seed_idx, mapped):
    """Bond indices that isolate `seed_idx`'s hanging region from the mapped scaffold.

    Always cuts direct bonds from the seed to other mapped atoms. If an alternate
    path remains (e.g. the other way around a ring), walks from the seed through
    *unmatched* atoms and cuts every edge that re-enters a mapped atom.
    """
    mapped = {int(i) for i in mapped}
    seed_idx = int(seed_idx)
    if seed_idx not in mapped:
        return []

    local = {seed_idx}
    stack = [seed_idx]
    cut = set()

    while stack:
        i = stack.pop()
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            if j in local:
                continue
            if j in mapped:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    cut.add(bond.GetIdx())
                continue
            local.add(j)
            stack.append(j)

    return sorted(cut)


def map_pendant_hydrogens(mol_ref, mol_query, mapping, df_ref, df_query):
    """Assign unmapped H on mapped heavies to free H on the partner ref heavy.

    Greedy by XYZ distance within each parent pair. Does not move pairs already
    in `mapping`. Returns (updated_mapping, soft_h_pairs) where soft_h_pairs
    are only the H assignments added here (safe for spatial_corrections to
    reorder — do not put them in overlap_mapping / locked_mapping).
    """
    mapping = dict(mapping)
    soft_h = {}
    occupied_r = set(mapping.values())
    ref_xyz = torch.tensor(df_ref.iloc[2:, 1:4].values.astype("float64"))
    query_xyz = torch.tensor(df_query.iloc[2:, 1:4].values.astype("float64"))

    parents = [
        (int(q), int(r))
        for q, r in list(mapping.items())
        if mol_query.GetAtomWithIdx(int(q)).GetSymbol() != "H"
        and mol_ref.GetAtomWithIdx(int(r)).GetSymbol() != "H"
    ]

    for q_parent, r_parent in parents:
        q_hs = [
            n.GetIdx()
            for n in mol_query.GetAtomWithIdx(q_parent).GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in mapping
        ]
        r_hs = [
            n.GetIdx()
            for n in mol_ref.GetAtomWithIdx(r_parent).GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in occupied_r
        ]
        if not q_hs or not r_hs:
            continue

        remaining_q = set(q_hs)
        remaining_r = set(r_hs)
        while remaining_q and remaining_r:
            best = None
            best_d = float("inf")
            for qh in remaining_q:
                for rh in remaining_r:
                    d = float(torch.norm(query_xyz[qh] - ref_xyz[rh]))
                    if d < best_d:
                        best_d = d
                        best = (qh, rh)
            if best is None:
                break
            qh, rh = best
            mapping[qh] = rh
            soft_h[qh] = rh
            occupied_r.add(rh)
            remaining_q.remove(qh)
            remaining_r.remove(rh)

    return mapping, soft_h


def apply_MCS(overlap_mapping, mol1, mol2, df_1, df_2, match_formal_charge=False):
    """Grow a query->ref map by MCS on fragments hanging off already-mapped atoms."""
    MCSandCoords_map = copy.deepcopy(overlap_mapping)
    mol1_atoms = mol1.GetAtoms()
    mol2_atoms = mol2.GetAtoms()

    for atom_idx in overlap_mapping:
        bonds = mol2_atoms[atom_idx].GetBonds()
        bonded_atoms = [
            bond.GetEndAtomIdx() if bond.GetEndAtomIdx() != atom_idx else bond.GetBeginAtomIdx()
            for bond in bonds
        ]
        matched_bonded_atoms = [a for a in bonded_atoms if a in MCSandCoords_map]
        if len(matched_bonded_atoms) == len(bonded_atoms):
            continue

        query_mapped = set(MCSandCoords_map)
        ref_mapped = set(MCSandCoords_map.values())
        bonds_to_break = bonds_to_cut_for_extension(mol2, atom_idx, query_mapped)
        if not bonds_to_break:
            continue

        ref_seed = MCSandCoords_map[atom_idx]
        ref_bonds_to_break = bonds_to_cut_for_extension(mol1, ref_seed, ref_mapped)
        if not ref_bonds_to_break:
            continue

        ref_f_mol, frag_ref_mapping = fragmented_to_mol(
            FragmentOnBonds(mol1, ref_bonds_to_break), ref_seed, dummies=False
        )
        query_f_mol, frag_query_mapping = fragmented_to_mol(
            FragmentOnBonds(mol2, bonds_to_break), atom_idx, dummies=False
        )
        ref_frag_mapping = {frag_ref_mapping[i]: i for i in frag_ref_mapping}
        query_frag_mapping = {frag_query_mapping[i]: i for i in frag_query_mapping}

        if atom_idx not in query_frag_mapping or ref_seed not in ref_frag_mapping:
            continue

        MCS_maps = produce_maps_list(
            ref_f_mol, query_f_mol, query_frag_mapping[atom_idx], ref_frag_mapping[ref_seed],
            match_formal_charge=match_formal_charge,
        )
        if MCS_maps == [{}]:
            continue

        # Among alternate fragment numberings, keep closest by XYZ proximity.
        MCS_map = choose_map(
            MCS_maps, frag_ref_mapping, frag_query_mapping, df_2, df_1
        )
        MCS_map = compare_to_previous(
            MCS_map, MCSandCoords_map, overlap_mapping, frag_query_mapping, frag_ref_mapping, df_2, df_1
        )

        for n in MCS_map:
            if (
                frag_query_mapping[n] not in overlap_mapping
                and frag_ref_mapping[MCS_map[n]] not in overlap_mapping.values()
                and frag_query_mapping[n] < len(mol2_atoms)
                and frag_ref_mapping[MCS_map[n]] < len(mol1_atoms)
            ):
                MCSandCoords_map = {
                    i: MCSandCoords_map[i]
                    for i in MCSandCoords_map
                    if (MCSandCoords_map[i] != frag_ref_mapping[MCS_map[n]] or i in overlap_mapping)
                }
                MCSandCoords_map[frag_query_mapping[n]] = frag_ref_mapping[MCS_map[n]]

    return MCSandCoords_map
