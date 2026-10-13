"""XYZ dataframe I/O, shuffling, and coordinate centering."""

import pandas as pd
import torch
import random

COORD_COLS = ["X", "Y", "Z"]


def get_centroid(df):
    """Geometric center (mean X, Y, Z) of atoms in an XYZ dataframe."""
    atoms = df.iloc[2:].copy()
    for col in COORD_COLS:
        atoms[col] = pd.to_numeric(atoms[col], errors="coerce")
    return atoms[COORD_COLS].mean()


def center_coords_df(df, centroid=None):
    """Return a copy with atom coordinates shifted so the centroid is at the origin."""
    df = df.copy()
    if centroid is None:
        centroid = get_centroid(df)
    for col in COORD_COLS:
        col_idx = df.columns.get_loc(col)
        df.iloc[2:, col_idx] = (
            pd.to_numeric(df.iloc[2:, col_idx], errors="coerce") - centroid[col]
        )
    return df


def center_coords_pair(df_ref, df_query):
    """Center reference and query using the reference centroid (preserves relative alignment)."""
    centroid = get_centroid(df_ref)
    return center_coords_df(df_ref, centroid), center_coords_df(df_query, centroid)


def get_batch_centroid(dfs):
    """Mean X/Y/Z over all atoms across multiple XYZ dataframes."""
    if not dfs:
        raise ValueError("At least one dataframe is required to compute a batch centroid")

    all_coords = []
    for df in dfs:
        atoms = df.iloc[2:].copy()
        for col in COORD_COLS:
            atoms[col] = pd.to_numeric(atoms[col], errors="coerce")
        all_coords.append(atoms[COORD_COLS])

    combined = pd.concat(all_coords, ignore_index=True)
    return combined.mean()


def get_batch_centroid_from_paths(paths):
    """Mean X/Y/Z over all atoms in a list of XYZ files."""
    return get_batch_centroid([xyz_to_df(path) for path in paths])


def xyz_to_df(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_atoms = int(lines[0].strip())

    # Extract atomic symbols and coordinates
    symbols = []
    coordinates = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) <= 1:
            break
        symbols.append(parts[0])
        coordinates.append([float(coord) for coord in parts[1:4]])

    # Create DataFrame
    df = pd.DataFrame({
        'Symbol': symbols,
        'X': [coord[0] for coord in coordinates],
        'Y': [coord[1] for coord in coordinates],
        'Z': [coord[2] for coord in coordinates]
    })
    
    new_row = pd.DataFrame({'Symbol': [len(symbols)], 'X': [''], 'Y': [''], 'Z': ['']})
    new_gap = pd.DataFrame({'Symbol': [''], 'X': [''], 'Y': [''], 'Z': ['']})
    df = pd.concat([new_row, new_gap, df], ignore_index=True)

    return df

def shuffle_df (df):
    
    coords = torch.tensor(df.iloc[2:,1:].values.astype('float64'))
    atoms = df.iloc[2:,:1].values
    
    val = list(range(coords.shape[0]))
    random.shuffle (val)
    coords = coords[val]
    atoms = atoms[val]
    
    df.iloc[2:,1:] = coords
    df.iloc[2:,:1] = atoms
    
    return df


def df_w_symbols(df, tensor, save=None):
    """Rebuild an XYZ dataframe from symbols in `df` and new coordinate rows.

    Resets indices before joining so atom rows from `iloc[2:]` (index 2,3,…)
    do not mis-align with a 0..n-1 coords frame and inflate the atom count.
    """
    symbols = df.iloc[2:, [0]].reset_index(drop=True)
    coords = pd.DataFrame(tensor.detach().numpy()).reset_index(drop=True)
    if len(symbols) != len(coords):
        raise ValueError(
            f"df_w_symbols: {len(symbols)} symbols vs {len(coords)} coordinate rows"
        )
    out = pd.concat([symbols, coords], axis=1)
    out.columns = ["Symbol", "X", "Y", "Z"]
    header = pd.DataFrame(
        {"Symbol": [len(symbols)], "X": [""], "Y": [""], "Z": [""]}
    )
    gap = pd.DataFrame({"Symbol": [""], "X": [""], "Y": [""], "Z": [""]})
    out = pd.concat([header, gap, out], ignore_index=True)

    if save is not None:
        print(save)
        out.to_csv(save + ".xyz", header=None, index=None, sep=" ", mode="w+")

    return out

def read_big_xyz_file(file_path):

    lst = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    num_atoms = int(lines[0].strip())
    length = len(lines)
    i = 2
    end_index = num_atoms
    while i < length:
        symbols = []
        coordinates = []
        for line in lines[i:end_index+2]:
            parts = line.strip().split()
            symbols.append(parts[0])
            coordinates.append([float(coord) for coord in parts[1:4]])
        df = pd.DataFrame({
            'Symbol': symbols,
            'X': [coord[0] for coord in coordinates],
            'Y': [coord[1] for coord in coordinates],
            'Z': [coord[2] for coord in coordinates]
            })
        i = i+num_atoms+2
        end_index = end_index + int(num_atoms) + 2
        new_row = pd.DataFrame({'Symbol': len(symbols),'X':'','Y': "",'Z': ''}, index=[0])
        new_gap = pd.DataFrame({'Symbol': '','X':'','Y': "",'Z': ''}, index=[0])
        df = pd.concat([new_row, new_gap, df])
        lst.append(df)
    return lst
        
   
