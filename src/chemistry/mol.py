"""Build RDKit molecules from XYZ dataframes."""

from _io import BytesIO

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDetermineBonds

from chemistry.xyz2mol import int_atom

CHARGES = [0, 1, -1, 2, -2, 3, -3, 4, -4]
charges_list = CHARGES


def draw_mol(mol):
    drawer = Draw.MolDraw2DCairo(512, 512)
    drawer.drawOptions().useBWAtomPalette()
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(BytesIO(drawer.GetDrawingText()))
    img.show()


def df_to_mol(df):
    """Convert an XYZ dataframe to an RDKit mol by trying common charges.

    Tries standard DetermineBonds first, then Hueckel-based bond ordering
    (needed for some heteroatom / halogen systems where the default fails).
    """
    xyz = df.iloc[:, :].to_string(header=False, index=False)
    attempts = (
        {"allowChargedFragments": True},
        {"allowChargedFragments": True, "useHueckel": True},
    )
    last_error = None
    for kwargs in attempts:
        for charge in CHARGES:
            try:
                mol = Chem.MolFromXYZBlock(xyz)
                if mol is None:
                    raise ValueError("MolFromXYZBlock returned None")
                rdDetermineBonds.DetermineBonds(mol, charge=charge, **kwargs)
                return mol
            except Exception as exc:
                last_error = exc
                continue
    raise ValueError(
        f"Could not perceive bonds from XYZ (tried charges {CHARGES} "
        f"with/without Hueckel). Last error: {last_error}"
    )


def atoms_and_coords_from_df(df):
    """Return atomic numbers and XYZ coordinates from an XYZ dataframe."""
    atoms = []
    xyz_coordinates = []
    for _, row in df.iloc[2:, :].iterrows():
        atoms.append(int_atom(row["Symbol"]))
        xyz_coordinates.append([row["X"], row["Y"], row["Z"]])
    return atoms, xyz_coordinates
