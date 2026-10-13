"""Geometric helpers used to seed alignment and filter mappings."""

import torch
from sklearn.decomposition import PCA


def find_best_fitted_surface(points):
    pca = PCA(n_components=2)
    pca.fit(points.numpy())
    return torch.tensor(pca.components_[0], dtype=torch.float32)


def rotation_angles_between_vectors(vec1, vec2):
    vec1 = vec1 / torch.norm(vec1)
    vec2 = vec2 / torch.norm(vec2)
    cross_product = torch.cross(vec1, vec2)
    yaw = torch.atan2(cross_product[0], cross_product[2])
    pitch = torch.asin(-cross_product[1])
    roll = torch.atan2(
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
        vec1[0] * vec2[0] + vec1[1] * vec2[1],
    )
    return torch.stack([pitch, yaw, roll])


def filter_lone(atom_mapping, sample_mol):
    """Drop mapped atoms that have no mapped bonded neighbor."""
    n_atoms = sample_mol.GetNumAtoms()
    # Ignore out-of-range keys (e.g. stale df/mol length mismatch).
    keys = [int(a) for a in atom_mapping if 0 <= int(a) < n_atoms]
    atoms_to_keep = []
    for atom1 in keys:
        if atom1 in atoms_to_keep:
            continue
        for atom2 in keys:
            if atom1 == atom2:
                continue
            if sample_mol.GetBondBetweenAtoms(atom1, atom2) is not None:
                atoms_to_keep += [atom1, atom2]
    return {i: atom_mapping[i] for i in keys if i in atoms_to_keep}
