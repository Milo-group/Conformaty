"""Fragment molecules around a chosen atom or bond."""

import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import FragmentOnBonds, GetMolFrags


def fragmented_to_mol(mol, main_idx, dummies=True):
    """Keep the fragment that contains `main_idx`; return (mol, new_idx -> old_idx)."""
    frags = GetMolFrags(mol)
    mw = Chem.RWMol(mol)

    atoms_to_remove = [a for f in frags if main_idx not in f for a in f]
    atoms_to_remove.sort()
    atoms_to_keep = [a for f in frags if main_idx in f for a in f]

    upt_idx = {j: j - len([i for i in atoms_to_remove if i < j]) for j in atoms_to_keep}

    for a_idx in atoms_to_remove[::-1]:
        for b in mw.GetAtoms()[a_idx].GetBonds():
            mw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

    for a_idx in atoms_to_remove[::-1]:
        mw.RemoveAtom(a_idx)

    for a in list(mw.GetAtoms())[::-1]:
        if a.GetSymbol() == "*":
            if mw.GetBondBetweenAtoms(upt_idx[main_idx], a.GetIdx()) is None or dummies is False:
                mw.RemoveAtom(a.GetIdx())
            else:
                mw.ReplaceAtom(a.GetIdx(), Chem.rdchem.Atom("*"))

    if dummies:
        mw.ReplaceAtom(upt_idx[main_idx], Chem.rdchem.Atom("*"))

    upt_idx = {upt_idx[i]: i for i in upt_idx}
    return mw.GetMol(), upt_idx


def even_fragments(mol):
    """Split `mol` on the bond that yields the most even fragment sizes."""
    mols_list = []
    gap_tensor = torch.tensor([])
    mol_atoms = mol.GetAtoms()

    for atom_idx in range(len(mol_atoms)):
        bonds = mol_atoms[atom_idx].GetBonds()
        bonds_to_break = [b.GetIdx() for b in bonds if b.GetIdx() > atom_idx]
        if not bonds_to_break:
            continue
        for bond in bonds_to_break:
            frag_mol = fragmented_to_mol(FragmentOnBonds(mol, [bond]), atom_idx, dummies=False)[0]
            if len(mol_atoms) - len(frag_mol.GetAtoms()) > 0:
                mols_list.append(bond)
                gap_tensor = torch.cat([
                    gap_tensor,
                    torch.abs(torch.tensor([len(mol_atoms) - 2 * len(frag_mol.GetAtoms())])),
                ])

    bond = mols_list[torch.argmin(gap_tensor)]
    begin_idx = mol.GetBonds()[bond].GetBeginAtomIdx()
    end_idx = mol.GetBonds()[bond].GetEndAtomIdx()
    return [
        fragmented_to_mol(FragmentOnBonds(mol, [bond]), begin_idx, dummies=False)[0],
        fragmented_to_mol(FragmentOnBonds(mol, [bond]), end_idx, dummies=False)[0],
    ]
