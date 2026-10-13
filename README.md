# Conformerity

Align two molecular structures and renumber the query atoms to match a reference.

## Installation

Python 3.10 or newer. From this folder:

```bash
pip install -r requirements.txt
```

That installs PyTorch, pandas, NumPy, RDKit, scikit-learn, Pillow, openpyxl, tqdm, NetworkX, and pyarrow.

The paper examples can be run from `test_cases/`. Open `test_cases/run_test_cases.ipynb` from this folder. Each case writes its results into that case's `output` folder.

## Usage

Run commands from this folder. `-t` is the reference and `-q` is the query. Pass `-o` to choose the output folder. Without it, results go under `outputs/`.

### Alignment

Alignment only moves the query as a rigid body. Atom numbers stay as they are. `--loss-type` is `mixed` (default), `rmsd`, `msd`, `mad`, or `sig`.

```bash
python main.py align -t test_cases/01_structural_alignment/cyclohexane_chair.xyz -q test_cases/01_structural_alignment/cyclohexane_boat.xyz --loss-type mixed -o test_cases/01_structural_alignment/output/mixed
```

### Renumbering

Renumbering also rewrites the query so its atoms follow the reference.

```bash
python main.py pair -t test_cases/02_prochirality/cyclohexanol.xyz -q test_cases/02_prochirality/cyclohexane_boat.xyz -o test_cases/02_prochirality/output
```

### Stages

Renumbering is modular. A short seed always runs first (MCS, rigid alignment, same-element proximity), and a short cleanup always runs last. Between them, `--stages` chooses which rules run and in which order. Any non-empty order of `complete_rings`, `extend_mcs`, and `coords_proximity_any` is valid. The default is all three, in that order.

One stage alone might be enough when that is the correspondence you want.

```bash
python main.py pair -t test_cases/03_ring_numbering/stage_order_ref.xyz -q test_cases/03_ring_numbering/stage_order_query.xyz --stages complete_rings -o test_cases/03_ring_numbering/output/complete_rings
```

### Anchors

An anchor is a pair of atoms that must correspond, even when the MCS would match them differently. On a single pair, pass 0-based indices. Several pairs are allowed; the two lists must be the same length. Case 04 uses this for the radical carbon of `ML2` and `ML275` (spreadsheet rows 2 and 5, which are indices 1 and 4). Those structures are from Zhang et al., *A neural network model informs the total synthesis of clovane sesquiterpenoids* (Nature Synthesis, 2023).

```bash
python main.py pair -t test_cases/04_radicals/ML2.xyz -q test_cases/04_radicals/ML275.xyz --match-charge --anchor-ref 1 --anchor-query 4 -o test_cases/04_radicals/output/pair_with_anchor
```

A batch can take the same information from an Excel file, one anchor per molecule against the reference. Case 04 uses `test_cases/04_radicals/Mol_RadNum.xlsx`. Two columns:

| Mol | idx |
| --- | --- |
| ML112 | 6 |
| ML275 | 5 |

`Mol` is the XYZ name without `.xyz`. `idx` is the anchored atom, counted from 1, as in the file. The reference must be a row too. Each query is then forced to pair its `idx` with the reference's `idx`.

```bash
python main.py batch -f test_cases/04_radicals -r test_cases/04_radicals/ML112.xyz --anchors-file test_cases/04_radicals/Mol_RadNum.xlsx --match-charge -o test_cases/04_radicals/output/batch_anchored
```

### Batch renumbering

`batch` renumbers every XYZ file in a folder onto one reference. `-f` is the folder and `-r` is the reference file inside it. The reference itself is skipped. Files in subfolders are not included. If `-r` is omitted, the smallest molecule in the folder is used. The same `--stages` and `--anchors-file` options apply to every structure.

Case 05 is a reaction pathway from Formica et al., *Catalytic enantioselective nucleophilic desymmetrization of phosphonate esters* (Nature Chemistry, 2023). The batch numbers the intermediate, transition states, and product onto the phosphonate substrate `SM.xyz`, so the whole mechanism shares one numbering.

```bash
python main.py batch -f test_cases/05_mechanism -r test_cases/05_mechanism/SM.xyz -o test_cases/05_mechanism/output/batch_vs_SM
```
