"""MCS-based atom renumbering: I/O orchestration around numbering stages."""

import os

import pandas as pd
from rdkit import Chem

from chemistry.mol import df_to_mol
from chemistry.rings import check_ring, find_init_vars_ring, ring_mapping
from fileio.paths import gen_out_path, gen_out_path_legacy, make_run_output_dir
from fileio.xyz import (
    center_coords_df,
    center_coords_pair,
    get_centroid,
    shuffle_df,
    xyz_to_df,
)
from chemistry.mcs import normalize_anchor_pairs
from numbering import RenumberingState, add_unmapped, run_numbering
from numbering.align import optimize_based_on_current_mapping
from numbering.spatial import spatial_corrections_map, spatial_matching_map


def _write_workflow(state, output_directory):
    path = os.path.join(output_directory, "numbering_workflow.txt")
    text = state.workflow_text()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    try:
        print(text.rstrip())
    except UnicodeEncodeError:
        print(text.rstrip().encode("ascii", "replace").decode("ascii"))
    print(f"Workflow written to: {path}")


def _maybe_center(df_1, df_2, center_coords, centroid, batch):
    if not center_coords:
        return df_1, df_2
    if centroid is not None:
        if not batch:
            df_1 = center_coords_df(df_1, centroid)
        df_2 = center_coords_df(df_2, centroid)
    elif batch:
        df_2 = center_coords_df(df_2, get_centroid(df_1))
    else:
        df_1, df_2 = center_coords_pair(df_1, df_2)
    return df_1, df_2


def _write_pair_xyz(df_1, df_2, file1_path, file2_path, output_directory, shuffle, batch):
    if not batch:
        target_suffix = "shuffled_target" if shuffle else "target"
        df_1.to_csv(
            gen_out_path(file1_path, target_suffix, output_directory),
            header=None, index=None, sep=" ", mode="w+",
        )
    df_2.to_csv(
        gen_out_path(file2_path, "optimized", output_directory),
        header=None, index=None, sep=" ", mode="w+",
    )


def renumber_pair(
    file1_path,
    file2_path,
    output_directory=None,
    anchors=None,
    shuffle=False,
    batch=False,
    center_coords=True,
    centroid=None,
    optimize_atoms=None,
    stage_order=None,
    debug=False,
    lr=None,
    max_iters=2000,
    match_formal_charge=False,
):
    """Renumber query atoms to match the target structure."""
    if output_directory is None:
        output_directory = make_run_output_dir(file2_path, "pair")
    os.makedirs(output_directory, exist_ok=True)

    df_1 = xyz_to_df(file1_path)
    df_2 = xyz_to_df(file2_path)

    # Keep the user-specified target when it is multi-fragment (sequential
    # matching needs the disconnected complex as ref). Otherwise prefer the
    # smaller structure as reference, as before.
    mol_target_probe = df_to_mol(df_1)
    n_target_frags = len(
        Chem.GetMolFrags(mol_target_probe, asMols=False, sanitizeFrags=False)
    )
    if n_target_frags <= 1 and df_1.shape[0] > df_2.shape[0]:
        df_1, df_2 = df_2, df_1
        file1_path, file2_path = file2_path, file1_path

    if shuffle:
        df_1 = shuffle_df(df_1)

    anchor_pairs = normalize_anchor_pairs(anchors)
    state = RenumberingState(
        df_ref=df_1,
        df_query=df_2,
        mol_ref=df_to_mol(df_1),
        mol_query=df_to_mol(df_2),
        optimize_atoms=optimize_atoms,
        anchor_pairs=anchor_pairs,
        lr=lr,
        max_iters=max_iters,
        match_formal_charge=match_formal_charge,
        debug=debug,
    )
    state = run_numbering(state, stage_order=stage_order, anchors=anchor_pairs)

    df_1, df_2 = _maybe_center(
        state.df_ref, state.df_query, center_coords, centroid, batch,
    )
    _write_pair_xyz(df_1, df_2, file1_path, file2_path, output_directory, shuffle, batch)
    if not batch:
        _write_workflow(state, output_directory)
    return state


renumbering = renumber_pair


def renumber_batch(
    folder_path, basic_path="", anchors_file=None, output_directory=None,
    center_coords=True, optimize_atoms=None, stage_order=None, debug=False,
    lr=None, max_iters=2000, match_formal_charge=False,
):
    """Renumber all XYZ files in a folder against a reference structure."""
    folder_path = os.path.abspath(folder_path)
    files_list = []
    min_size = 0
    min_path = None
    invalid_files = []
    anchors_dic = None

    if anchors_file is not None:
        anchors_df = pd.read_excel(anchors_file)
        anchors_dic = {
            os.path.join(folder_path, f"{anchors_df['Mol'][i]}.xyz"): anchors_df["idx"][i] - 1
            for i in range(anchors_df.shape[0])
        }

    basic_size = None
    if basic_path:
        basic_path = os.path.abspath(basic_path)
        if not os.path.isfile(basic_path):
            raise FileNotFoundError(f"Reference file not found: {basic_path}")
        df = xyz_to_df(basic_path)
        mol = df_to_mol(df)
        if not mol:
            raise ValueError(f"Could not determine valid charge for reference: {basic_path}")
        basic_size = len(mol.GetAtoms())

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".xyz"):
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            df = xyz_to_df(file_path)
            mol = df_to_mol(df)
        except Exception:
            print(f"{file_name} is not valid")
            invalid_files.append(file_name)
            continue
        if not mol:
            print(f"{file_name} charge is not valid")
            invalid_files.append(file_name)
            continue
        if basic_path:
            if len(mol.GetAtoms()) < basic_size:
                raise ValueError(f"{file_name} has fewer atoms than the reference")
        elif min_size == 0 or len(mol.GetAtoms()) < min_size:
            min_path = file_path
            min_size = len(mol.GetAtoms())
        files_list.append(file_path)

    if not files_list:
        raise ValueError("No valid XYZ files found in folder")
    if not basic_path:
        basic_path = min_path

    if output_directory is None:
        output_directory = make_run_output_dir(folder_path, "batch")
    os.makedirs(output_directory, exist_ok=True)

    df_basic = xyz_to_df(basic_path)
    batch_centroid = None
    if center_coords:
        batch_centroid = get_centroid(df_basic)
        print(
            "Batch centering: using the reference centroid "
            f"({batch_centroid['X']:.4f}, {batch_centroid['Y']:.4f}, {batch_centroid['Z']:.4f})"
        )
        df_basic = center_coords_df(df_basic, batch_centroid)

    df_basic.to_csv(
        gen_out_path(basic_path, "target", output_directory),
        header=None, index=None, sep=" ", mode="w+",
    )
    print(f"Reference structure: {basic_path}")
    print(f"Output folder: {output_directory}")

    error_files = []
    for idx, file_path in enumerate(files_list):
        print(f"Processing {file_path} ({idx + 1}/{len(files_list)})")
        if file_path == basic_path:
            print("Skipping reference file")
            continue
        try:
            anchors = None
            if anchors_dic is not None:
                # One forced pair from Excel (ref_idx, query_idx), 0-based.
                anchors = [(anchors_dic[basic_path], anchors_dic[file_path])]
            renumber_pair(
                basic_path, file_path, output_directory,
                anchors=anchors, batch=True,
                center_coords=center_coords, centroid=batch_centroid,
                optimize_atoms=optimize_atoms,
                stage_order=stage_order,
                debug=debug,
                lr=lr,
                max_iters=max_iters,
                match_formal_charge=match_formal_charge,
            )
        except Exception as exc:
            print(f"Failed: {file_path} ({exc})")
            error_files.append(file_path)

    print("Batch renumbering finished.")
    print(f"Invalid files ({len(invalid_files)}): {invalid_files}")
    print(f"Failed files ({len(error_files)}): {error_files}")
    return output_directory


batch_renumbering = renumber_batch


def _check_mul_ring(atom_map, rings):
    for ring in rings:
        if check_ring(atom_map, ring):
            return True
    return False


def _check_bond(atom_map, ring1, ring2):
    if _check_mul_ring(list(atom_map.keys()), ring1) and _check_mul_ring(list(atom_map.values()), ring2):
        flag = 0
        keys = list(atom_map.keys())
        vals = list(atom_map.values())
        for item in ring1:
            if (keys[0] in item and keys[1] in item) or (vals[0] in item and vals[1] in item):
                flag += 1
                for item2 in ring2:
                    if (keys[0] in item2 and keys[1] in item2) or (vals[0] in item2 and vals[1] in item2):
                        flag += 1
        if flag == 2:
            return True
        if flag >= 2:
            atom1 = input("Please enter a different atom on the ring from molecule #1: ")
            atom2 = input("Please enter a different atom on the ring from molecule #2: ")
            if atom1.isnumeric() and atom2.isnumeric():
                return [int(atom1) - 1, int(atom2) - 1]
            return False
        return False
    return False


def renumber_ring(
    file1_path,
    file2_path,
    output_directory="molecules/results",
    shuffle=False,
    batch=False,
    temp_map=None,
):
    """Renumber using a known bond anchor on a ring system."""
    if temp_map is None:
        temp_map = {0: 0, 1: 1}

    os.makedirs(output_directory, exist_ok=True)
    df_1 = xyz_to_df(file1_path)
    df_2 = xyz_to_df(file2_path)
    swapped = False

    if df_1.shape[0] > df_2.shape[0]:
        swapped = True
        df_1, df_2 = df_2, df_1
        file1_path, file2_path = file2_path, file1_path
        keys = list(temp_map.keys())
        vals = list(temp_map.values())
        temp_map = {vals[0]: keys[0], vals[1]: keys[1]}

    if shuffle:
        df_1 = shuffle_df(df_1)

    mol1 = df_to_mol(df_1)
    mol2 = df_to_mol(df_2)
    ring1 = [[i for i in ring] for ring in Chem.GetSSSR(mol1)]
    ring2 = [[i for i in ring] for ring in Chem.GetSSSR(mol2)]

    checkbond = _check_bond(temp_map, ring1, ring2)
    if not checkbond:
        print("Error: please choose a different bond")
        return False

    if swapped:
        keys = list(temp_map.keys())
        vals = list(temp_map.values())
        temp_map = {vals[0]: keys[0], vals[1]: keys[1]}

    init_vars = find_init_vars_ring(df_1, df_2, temp_map, checkbond)
    tmap = dict(temp_map)
    df_2, init_vars = optimize_based_on_current_mapping(df_1, df_2, temp_map, mol2, init_vars)
    temp_map = ring_mapping(tmap, mol1, mol2, checkbond)
    atom_map = dict(temp_map)
    atom_map.update(spatial_corrections_map(temp_map, atom_map, mol2, df_1, df_2))
    atom_map.update(spatial_matching_map(atom_map, mol2, mol1, df_2, df_1))
    atom_map.update(add_unmapped(atom_map, mol2))

    final_map = {atom_map[i]: i for i in range(len(atom_map))}
    final_order = [final_map[i] for i in range(len(final_map))]
    for i in range(len(final_order)):
        if i not in final_order:
            final_order.append(i)

    df_t = df_2.iloc[:2, :]
    df_2 = df_2.iloc[2:, :].iloc[final_order, :]
    df_2 = pd.concat([df_t, df_2])

    if not batch:
        target_suffix = "shuffled_target" if shuffle else "target"
        df_1.to_csv(
            gen_out_path_legacy(file1_path, target_suffix, output_directory),
            header=None, index=None, sep=" ", mode="w+",
        )
    df_2.to_csv(
        gen_out_path_legacy(file2_path, "optimized", output_directory),
        header=None, index=None, sep=" ", mode="w+",
    )
    return True


renumbering_ring = renumber_ring
