"""Spatial alignment pipeline without atom renumbering."""

import os
from glob import glob

import pandas as pd
import torch
from rdkit.Chem.rdmolops import FragmentOnBonds

from chemistry.fragments import fragmented_to_mol
from chemistry.identities import check_identities, check_order_prochiral
from chemistry.mcs import find_init_vars
from chemistry.mol import df_to_mol
from fileio.paths import gen_out_path, make_run_output_dir
from fileio.xyz import center_coords_df, center_coords_pair, get_centroid, xyz_to_df
from optimization.optimizer import optimize_numbered, resolve_optimize_atom_pairs


def _correct_order_prochiral(new_order, existing_order, temp_map, update_map):
    for idx, n in enumerate(new_order):
        init_dic = existing_order[list(existing_order.keys())[idx]]
        target_dic = existing_order[n]

        if init_dic == target_dic or len(init_dic) != len(target_dic):
            continue

        for i in init_dic:
            if init_dic[i] in temp_map and target_dic[i] in temp_map:
                update_map[init_dic[i]] = temp_map[target_dic[i]]

    return update_map


def prochiral_corrections(temp_map, mol_ref, df_1, df_2):
    update_map = {}
    mol_ref_atoms = mol_ref.GetAtoms()

    for atom_idx in temp_map:
        bonds = mol_ref_atoms[atom_idx].GetBonds()
        if len(bonds) < 4:
            continue

        bonded_atoms = [
            bond.GetEndAtomIdx() if bond.GetEndAtomIdx() != atom_idx else bond.GetBeginAtomIdx()
            for bond in bonds
        ]
        mols_to_compare = []

        for r in bonded_atoms:
            bonds_to_break = [
                b.GetIdx() for b in bonds if (b.GetEndAtomIdx() != r and b.GetBeginAtomIdx() != r)
            ]
            mols_to_compare.append(fragmented_to_mol(FragmentOnBonds(mol_ref, bonds_to_break), atom_idx))

        identical_substituents = check_identities(mols_to_compare, bonded_atoms)
        participating_atoms = [atom_idx] + bonded_atoms
        ord_identical_substituents = [
            check_order_prochiral(s, temp_map, df_2, df_1, participating_atoms) if len(s) > 1 else []
            for s in identical_substituents
        ]

        for idx in range(len(ord_identical_substituents)):
            if (
                ord_identical_substituents[idx] != list(identical_substituents[idx].keys())
                and ord_identical_substituents[idx] != []
            ):
                update_map = _correct_order_prochiral(
                    ord_identical_substituents[idx], identical_substituents[idx], temp_map, update_map
                )

    return update_map


def _compute_rmsd(df_ref, df_query, target_atoms, query_atoms=None):
    if query_atoms is None:
        query_atoms = target_atoms

    ref_coords = torch.tensor(df_ref.iloc[2:, 1:4].values.astype("float64")).to(torch.float64)
    opt_coords = torch.tensor(df_query.iloc[2:, 1:4].values.astype("float64")).to(torch.float64)

    ref_coords = ref_coords[target_atoms]
    opt_coords = opt_coords[query_atoms]

    squared_diff = torch.sum(torch.pow(ref_coords - opt_coords, 2), dim=1)
    return torch.sqrt(torch.mean(squared_diff)).item()


def _initial_alignment_vars(df_ref, df_query, match_formal_charge=False, with_hs=False):
    """Return (init_vars, mcs_map): PCA/centroid init plus the query->ref MCS map."""
    mol_ref = df_to_mol(df_ref)
    mol_query = df_to_mol(df_query)
    init_vars = find_init_vars(
        mol_ref, mol_query, df_ref, df_query,
        match_formal_charge=match_formal_charge,
        with_hs=with_hs,
    )
    return init_vars, init_vars[3][0]


def align_pair(
    ref_path,
    query_path,
    output_directory=None,
    loss_type="mixed",
    sigma=30,
    lr=None,
    max_iters=2000,
    optimize_atoms=None,
    center_coords=True,
    centroid=None,
    match_formal_charge=False,
    debug=False,
    with_hs=False,
):
    """Align query coordinates onto the reference without renumbering atoms."""
    if output_directory is None:
        output_directory = make_run_output_dir(query_path, "align")
    os.makedirs(output_directory, exist_ok=True)

    df_ref = xyz_to_df(ref_path)
    df_query = xyz_to_df(query_path)

    if optimize_atoms is not None:
        n_atoms = len(df_ref) - 2
        invalid = [i for i in optimize_atoms if i < 0 or i >= n_atoms]
        if invalid:
            raise ValueError(f"Reference atom indices out of range (0-{n_atoms - 1}): {invalid}")

    init_vars, temp_map = _initial_alignment_vars(
        df_ref, df_query, match_formal_charge=match_formal_charge, with_hs=with_hs,
    )
    target_atoms, query_atoms = resolve_optimize_atom_pairs(temp_map, optimize_atoms)

    print("Aligning structures...")
    df_query = optimize_numbered(
        df_ref, df_query, target_atoms, query_atoms=query_atoms,
        init_vars=init_vars, lr=lr, loss_type=loss_type, sigma=sigma,
        max_iters=max_iters, debug=debug,
    )

    if center_coords:
        if centroid is not None:
            df_ref = center_coords_df(df_ref, centroid)
            df_query = center_coords_df(df_query, centroid)
        else:
            df_ref, df_query = center_coords_pair(df_ref, df_query)

    df_ref.to_csv(
        gen_out_path(ref_path, "target", output_directory),
        header=None, index=None, sep=" ", mode="w+",
    )
    df_query.to_csv(
        gen_out_path(query_path, "optimized", output_directory),
        header=None, index=None, sep=" ", mode="w+",
    )

    rmsd = _compute_rmsd(df_ref, df_query, target_atoms, query_atoms)
    print(f"RMSD: {rmsd:.4f} Angstrom")
    print(f"Output saved to: {output_directory}")
    return rmsd


align = align_pair


def _choose_min_atoms_ref(folder_xyz_files):
    min_atoms = None
    chosen_path = None

    for path in folder_xyz_files:
        try:
            df = xyz_to_df(path)
            n_atoms = max(len(df) - 2, 0)
        except Exception as exc:
            print(f"Warning: failed to read '{path}' ({exc}), skipping.")
            continue

        if min_atoms is None or n_atoms < min_atoms:
            min_atoms = n_atoms
            chosen_path = path

    return chosen_path


def align_batch(
    folder_path,
    ref_path=None,
    output_directory=None,
    loss_type="mixed",
    sigma=30,
    lr=None,
    max_iters=2000,
    optimize_atoms=None,
    center_coords=True,
    debug=False,
):
    """Align all XYZ files in a folder to a single reference."""
    folder_path = os.path.abspath(folder_path)
    folder_xyz_files = sorted(glob(os.path.join(folder_path, "*.xyz")))

    if not folder_xyz_files:
        raise ValueError(f"No .xyz files found in: {folder_path}")

    if ref_path is None:
        ref_path = _choose_min_atoms_ref(folder_xyz_files)
        if ref_path is None:
            raise ValueError("Could not determine a reference structure from folder.")
        print(f"Using smallest molecule as reference: {ref_path}")
    else:
        ref_path = os.path.abspath(ref_path)

    if output_directory is None:
        output_directory = make_run_output_dir(folder_path, "align")
    os.makedirs(output_directory, exist_ok=True)

    batch_centroid = None
    if center_coords:
        batch_centroid = get_centroid(xyz_to_df(ref_path))
        print(
            "Batch centering: using the reference centroid "
            f"({batch_centroid['X']:.4f}, {batch_centroid['Y']:.4f}, {batch_centroid['Z']:.4f})"
        )

    print(f"Batch alignment: {len(folder_xyz_files)} files in '{folder_path}'")
    print(f"Output directory: {output_directory}")

    success_count = 0
    failed_files = []
    rmsd_results = []

    for query_path in folder_xyz_files:
        if os.path.abspath(query_path) == os.path.abspath(ref_path):
            continue

        print(f"\nAligning: {query_path}")
        try:
            rmsd = align_pair(
                ref_path, query_path,
                output_directory=output_directory,
                loss_type=loss_type, sigma=sigma, lr=lr, max_iters=max_iters,
                optimize_atoms=optimize_atoms,
                center_coords=center_coords, centroid=batch_centroid,
                debug=debug,
            )
            success_count += 1
            rmsd_results.append({"filename": os.path.basename(query_path), "rmsd": rmsd})
        except Exception as exc:
            print(f"Failed: {query_path} ({exc})")
            failed_files.append(query_path)

    if rmsd_results:
        csv_path = os.path.join(output_directory, "rmsd_results.csv")
        pd.DataFrame(rmsd_results).to_csv(csv_path, index=False)
        print(f"\nRMSD results saved to: {csv_path}")

    print("\nBatch alignment summary:")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {len(failed_files)}")
    if failed_files:
        for path in failed_files:
            print(f"    - {path}")

    return output_directory
