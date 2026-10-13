"""Ring-aware mapping helpers."""

import copy

import torch
from rdkit import Chem

from chemistry.geometry import find_best_fitted_surface, rotation_angles_between_vectors
from chemistry.mol import df_to_mol


def check_ring(atom_map, ring1):
    leng = len(ring1) - 1
    indx_1_1 = 0
    indx_1_2 = 0
    if atom_map[0] in ring1:
        indx_1_1 = ring1.index(atom_map[0])
    if atom_map[1] in ring1:
        indx_1_2 = ring1.index(atom_map[1])
    return (
        abs(indx_1_1 - indx_1_2) == 1
        or (indx_1_1 == 0 and indx_1_2 == leng)
        or (indx_1_1 == leng and indx_1_2 == 0)
    )


def find_init_vars_ring(file_path1, file_path2, map, atom3):
    query_mol = df_to_mol(file_path1)
    ref_mol = df_to_mol(file_path2)

    ring_ref = [[i for i in ring] for ring in Chem.GetSSSR(ref_mol)]
    ring_query = [[i for i in ring] for ring in Chem.GetSSSR(query_mol)]

    for itm in ring_ref:
        if list(map.keys())[0] in itm and list(map.keys())[1] in itm and (atom3 is True or atom3[0] in itm):
            ring_ref = itm
            break

    for itm in ring_query:
        if list(map.values())[0] in itm and list(map.values())[1] in itm and (atom3 is True or atom3[1] in itm):
            ring_query = itm
            break

    df_2_tensor = torch.tensor(file_path2.iloc[2:, 1:].values.astype("float64"))
    df_1_tensor = torch.tensor(file_path1.iloc[2:, 1:].values.astype("float64"))
    COM_2 = torch.sum(df_2_tensor[ring_query], 0) / len(ring_query)
    COM_1 = torch.sum(df_1_tensor[ring_ref], 0) / len(ring_ref)
    PCA2 = find_best_fitted_surface(df_2_tensor[ring_query])
    PCA1 = find_best_fitted_surface(df_1_tensor[ring_ref])
    PCA_angles = rotation_angles_between_vectors(PCA1, PCA2)

    from optimization.rotations import rot

    df_2_tensor_rotated = rot(df_2_tensor.type(torch.FloatTensor), PCA_angles[0], PCA_angles[1], PCA_angles[2])
    return df_1_tensor - COM_1, df_2_tensor_rotated - COM_2


def ring_mapping(overlap_mapping, mol1, mol2, atom3):
    mol1_atoms = mol1.GetAtoms()
    mol2_atoms = mol2.GetAtoms()
    ring1 = [[i for i in ring] for ring in Chem.GetSSSR(mol1)]
    ring2 = [[i for i in ring] for ring in Chem.GetSSSR(mol2)]
    keys = list(overlap_mapping.keys())
    vals = list(overlap_mapping.values())

    for itm in ring1:
        if vals[0] in itm and vals[1] in itm and (atom3 is True or atom3[1] in itm):
            ring1 = itm
            break

    for itm in ring2:
        if keys[0] in itm and keys[1] in itm and (atom3 is True or atom3[0] in itm):
            ring2 = itm
            break

    ring1_len = len(ring1)
    ring2_len = len(ring2)
    if ring1_len == ring2_len:
        while ring2[0] != list(overlap_mapping.keys())[0]:
            ring2 = ring2[-1:] + ring2[:-1]
        while ring1[0] != list(overlap_mapping.values())[0]:
            ring1 = ring1[-1:] + ring1[:-1]
        mapping = copy.deepcopy(overlap_mapping)
        for i in range(len(ring1) // 2):
            if ring2[i] not in list(mapping.keys()) and ring1[i] not in list(mapping.values()):
                mapping[ring2[i]] = ring1[i]
        for i in range(len(ring2) - 1, len(ring1) // 2 - 1, -1):
            if ring2[i] not in list(mapping.keys()) and ring1[i] not in list(mapping.values()):
                mapping[ring2[i]] = ring1[i]
    else:
        if ring1_len < ring2_len:
            ring1, ring2 = ring2, ring1
            ring1_len, ring2_len = ring2_len, ring1_len
            mol1_atoms, mol2_atoms = mol2_atoms, mol1_atoms
        while ring1[0] not in list(overlap_mapping.keys()):
            ring1 = ring1[-1:] + ring1[:-1]
        if ring1[0] in ring2:
            while ring2[0] != ring1[0]:
                ring2 = ring2[-1:] + ring2[:-1]
        else:
            while ring2[0] not in list(overlap_mapping.values()):
                ring2 = ring2[-1:] + ring2[:-1]
        mapping = copy.deepcopy(overlap_mapping)
        ring1 = ring1[2:]
        ring2 = ring2[2:]
        for i in range(len(ring2) // 2):
            if ring1[i] not in list(mapping.keys()) and ring2[i] not in list(mapping.values()):
                mapping[ring1[i]] = ring2[i]
        j = len(ring1) - 1
        for i in range(len(ring2) - 1, len(ring2) // 2 - 1, -1):
            if ring1[i] not in list(mapping.keys()) and ring2[i] not in list(mapping.values()):
                mapping[ring1[j]] = ring2[i]
            j = j - 1

    temp_map = copy.deepcopy(mapping)
    print(ring1)
    print(ring2)
    bonded_atoms1 = []
    bonded_atoms2 = []
    for atom_idx in list(temp_map.keys()):
        bonds1 = mol1_atoms[atom_idx].GetBonds()
        bonded_atoms1.append([
            bond.GetEndAtomIdx() if bond.GetEndAtomIdx() != atom_idx else bond.GetBeginAtomIdx()
            for bond in bonds1
        ])
    for atom_idx in list(temp_map.values()):
        bonds2 = mol2_atoms[atom_idx].GetBonds()
        bonded_atoms2.append([
            bond.GetEndAtomIdx() if bond.GetEndAtomIdx() != atom_idx else bond.GetBeginAtomIdx()
            for bond in bonds2
        ])
    for k in range(len(bonded_atoms1)):
        for j in bonded_atoms2[k]:
            for i in bonded_atoms1[k]:
                if i not in list(mapping.keys()) and j not in list(mapping.values()) and i not in ring1:
                    mapping[i] = j
                    break
    return mapping
