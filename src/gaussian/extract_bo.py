import pandas as pd
import re
from rdkit import Chem

#  Parse the log file
log_file_path = "m_Br.log"

def log_file_to_bo(log_file_path):
    #extract atom ix, # bonds and #lone pairs and create a dictionary: {atom ix:[number of bonds, numer of lp]}
    bd_data = []  # List to store BD lines
    lp_data = []  # List to store LP lines
    start_parsing = False  # Flag to indicate when parsing should start

    with open(log_file_path, "r") as file:
        for line in file:
            # Check if the starting line is encountered
            if "natural bond orbitals (summary):" in line.lower().strip():
                print("Starting to parse after encountering 'Natural Bond Orbitals (Summary):'")

                start_parsing = True
                continue  # Skip the current line and move to the next

            if start_parsing:
                # Assuming you're iterating over the lines
                if "BD" in line or "LP" in line:
                    if "BD" in line:
                        start_index = line.find("BD")
                        bd_data.append(line[start_index:].strip())  # Add to BD list
                    elif "LP" in line:
                        start_index = line.find("LP")
                        lp_data.append(line[start_index:].strip())  # Add to LP list
                elif "CR" in line.strip():
                    continue
                elif "RY" in line.strip():  # End condition
                    print("Stopping parsing after encountering 'RY'")
                    break

    # Create DataFrames for BD and LP
    df_bd = pd.DataFrame(bd_data, columns=["Content"])
    df_lp = pd.DataFrame(lp_data, columns=["Content"])

    # Reset index for each DataFrame
    df_bd = df_bd.reset_index(drop=True)
    df_lp = df_lp.reset_index(drop=True)

    #extract numbers
    df_bd['Numbers_BD'] = df_bd['Content'].apply(lambda x: re.findall(r"[-+]?\d*\.\d+|\d+", x))
    df_lp['Numbers_LP'] = df_lp['Content'].apply(lambda x: re.findall(r"[-+]?\d*\.\d+|\d+", x))

    atom_bonds = [] #this list contains two atoms and the bond between them
    mol_atoms = [] #this list contains all the atoms and the finel bond order

    # Iterate over the 'Numbers_BD' column
    for index, row in df_bd.iterrows():
        # Access the list in the 'Numbers_BD' column
        numbers = row['Numbers_BD']

        # Ensure the numbers list has enough elements before proceeding
        if len(numbers) >= 3:
            atom_bonds = [int(numbers[1]), int(numbers[2]), int(numbers[0])]  # Creating a new bond list for each row

            if len(mol_atoms) != 0:
                # double or triple bonds - Check if the current atom bonds match the previous one in 'mol_atoms'
                if atom_bonds[1] == mol_atoms[-1][1] and atom_bonds[0] == mol_atoms[-1][0]:
                    mol_atoms[-1] = atom_bonds  # Replace the last entry
                else:
                    mol_atoms.append(atom_bonds)  # Append only if no replacement occurs
            else:
                # If mol_atoms is empty, append the first entry
                mol_atoms.append(atom_bonds)

    #dictionary with the number of bonds for each atom
    bo_dic = {}
    for bond in mol_atoms:
        atom1, atom2, bond_count = bond

        # Update bond count for the first atom
        if atom1 in bo_dic:
            bo_dic[atom1] += bond_count
        else:
            bo_dic[atom1] = bond_count

        # Update bond count for the second atom
        if atom2 in bo_dic:
            bo_dic[atom2] += bond_count
        else:
            bo_dic[atom2] = bond_count


    #add number of lone pairs to the BO dictionary
    for index, row in df_lp.iterrows():
        numbers_lp = row['Numbers_LP']
        key = int(numbers_lp[1])
        lone_p = int(numbers_lp[0])
        if isinstance(bo_dic[key], tuple):
            current_bonds = bo_dic[key][0]
            bo_dic[key] = (current_bonds, lone_p)  # Correct update if it's a tuple
        else:
            bo_dic[key] = (bo_dic[key], lone_p)

    #add 0 lone pairs

    for key, value in bo_dic.items():
        if not isinstance(value, tuple):
            bo_dic[key] = (value, 0)

    return bo_dic, mol_atoms

def xyz_to_symbol_list(xyz_df):

    if 'Symbol' not in xyz_df.columns:
        raise ValueError("DataFrame must contain an 'atom' column.")
    
    symbols = xyz_df['Symbol'].tolist()[2:]  # Skip the first two rows

    return symbols

atomic_valence_electrons = {}
atomic_valence_electrons[1] = 1
atomic_valence_electrons[5] = 3
atomic_valence_electrons[6] = 4
atomic_valence_electrons[7] = 5
atomic_valence_electrons[8] = 6
atomic_valence_electrons[9] = 7
atomic_valence_electrons[14] = 4
atomic_valence_electrons[15] = 5
atomic_valence_electrons[16] = 6
atomic_valence_electrons[17] = 7
atomic_valence_electrons[32] = 4
atomic_valence_electrons[35] = 7
atomic_valence_electrons[53] = 7

global __ATOM_LIST__
__ATOM_LIST__ = \
    ['h',  'he',
     'li', 'be', 'b',  'c',  'n',  'o',  'f',  'ne',
     'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',
     'k',  'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
     'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
     'rb', 'sr', 'y',  'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
     'cd', 'in', 'sn', 'sb', 'te', 'i',  'xe',
     'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
     'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt',
     'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
     'fr', 'ra', 'ac', 'th', 'pa', 'u',  'np', 'pu']



def get_atomic_number(xyz_df, BO_dic):
    #add the atom symbol to the dictionary: {atom ix: [symbol,number of bonds]}

    symbols = xyz_to_symbol_list(xyz_df)

    # Iterate over the atoms in the bo_dic dictionary
    for atom in BO_dic.keys():
        symbol = symbols[atom-1]  # Get the atomic symbol for the atom
        atomic_number = __ATOM_LIST__.index(symbol.lower())+1
        current_bonds_lone_pairs = BO_dic[atom]

        #add the atomic numbre to the value tuple
        BO_dic[atom] = current_bonds_lone_pairs + (atomic_number,)

    return BO_dic

def get_formal_charge(BO_dic, atomic_valence_electrons):
    #add formal charge based on valence, number of bonds and lone pairs
    for atom, value in BO_dic.items():
        bonds = value[0]
        lone_pairs = value[1]
        valence = atomic_valence_electrons[value[2]]
        charge = valence-lone_pairs*2-bonds
        current_value = BO_dic[atom]

        # add the atomic numbre to the value tuple
        BO_dic[atom] = current_value + (charge,)

        continue
    return BO_dic

from rdkit import Chem

def build_a_mol(BO_dic, mol_atoms):
    bond_order_dict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    # Initialize an EditableMol object
    editable_mol = Chem.EditableMol(Chem.Mol())
    # Add atoms to the molecule
    atom_mapping = {}  # Maps BO_dic indices to molecule atom indices
    for atom_idx, value in BO_dic.items():
        atomic_number = value[2]  # Atomic number
        formal_charge = value[3]  # Formal charge
        atom = Chem.Atom(atomic_number)
        atom.SetFormalCharge(formal_charge)
        mol_idx = editable_mol.AddAtom(atom)
        atom_mapping[atom_idx] = mol_idx  # Map BO_dic index to RDKit atom index

    # Add bonds to the molecule
    for bond in mol_atoms:
        ix1 = bond[0]  # Atom 1 index in BO_dic
        ix2 = bond[1]  # Atom 2 index in BO_dic
        bond_order = bond[2]  # Bond order
        bond_type = bond_order_dict[bond_order]  # Map bond order to RDKit bond type
        editable_mol.AddBond(atom_mapping[ix1], atom_mapping[ix2], bond_type)

    # Convert the EditableMol to a Mol object
    mol = editable_mol.GetMol()

    from rdkit.Chem import MolToSmiles
    return mol  # Return the sanitized RDKit molecule


def log_xyz_to_mol(xyz_df, log_file):
    BO_dic, mol_atoms = log_file_to_bo(log_file)
    Bo_dic = get_atomic_number(xyz_df, BO_dic)
    BO_dic = get_formal_charge(BO_dic, atomic_valence_electrons)
    mol = build_a_mol(BO_dic, mol_atoms)
    return mol
