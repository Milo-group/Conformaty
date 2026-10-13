"""Rigid-body coordinate optimization of a query onto a reference."""

import math

import torch
from torch.autograd import Variable

from fileio.xyz import df_w_symbols
from optimization.loss import loss_function
from optimization.rotations import rotate_and_move

def analyze_df (df, atoms_dictionary = {}):

    coords_tensor = torch.tensor(df.iloc[2:,1:].values.astype('float64'))

    atoms_list = []

    # Processing of the target and query is performed diffrently.
    # For the target, the goal is to create an atom dictionary for the one-hot embedding (e.g - C:0, N:1, H:2, etc.).
    # For the query, we would like to use the same dictionary as used for the target.
    # If there's an atom in the query that is not part of the target, it would be assigned to the last class, which
    # contains all query atom types that are not in the part of the target molecules 

    if atoms_dictionary == {}:
        target_flag = True
    else:
        target_flag = False
    
    # print (df)
    #iterate over all atoms in the dataframe
    for atom in df.iloc[2:,:1].values:
        atom = atom[0]

        if target_flag:
            if atom not in atoms_dictionary:            
                atoms_dictionary[atom] = len(atoms_dictionary)
            atoms_list.append(atoms_dictionary[atom])

        else:
            if atom not in atoms_dictionary:
                atoms_list.append(len(atoms_dictionary)) 
            else:
                atoms_list.append(atoms_dictionary[atom])
        
    atoms_tensor = torch.nn.functional.one_hot(torch.tensor(atoms_list), num_classes=len(atoms_dictionary)+1)         
    full_tensor = torch.concat([coords_tensor, atoms_tensor], dim = 1)

    return coords_tensor, atoms_tensor, full_tensor, atoms_dictionary

def init_random():

    initial_vars = torch.rand(6).to(torch.float64)
    initial_vars[:3] *= (2 * torch.pi)
    initial_vars[3:] *= 2
    
    return [Variable(initial_vars, requires_grad=True)]

def init_angles(Angles, init_vars):

    initial_vars = torch.zeros(6).to(torch.float64)
    initial_vars [3] = init_vars[2][0] + Angles['X_angle']
    initial_vars [4] = init_vars[2][1] + Angles['Y_angle']
    initial_vars [5] = init_vars[2][2] + Angles['Z_angle']

    return [Variable(initial_vars, requires_grad=True)]



def Overlap_Mapping(target_df, query_df, query_mol, dist_threshold=None):
    """Backward-compatible alias for numbering.spatial.overlap_mapping."""
    from numbering.spatial import overlap_mapping as _overlap_mapping
    return _overlap_mapping(target_df, query_df, query_mol, dist_threshold=dist_threshold)

def resolve_optimize_atom_pairs(temp_map, optimize_atoms=None):
    """Map reference atom indices to paired (target, query) indices via MCS.

    temp_map maps query_idx -> ref_idx (as returned by find_init_vars / apply_MCS).
    optimize_atoms are 0-based row indices in the reference XYZ (first atom = 0).
    """
    if optimize_atoms is not None:
        target_atoms = list(optimize_atoms)
    else:
        target_atoms = sorted(set(temp_map.values()))

    ref_to_query = {ref_idx: query_idx for query_idx, ref_idx in temp_map.items()}
    missing = [idx for idx in target_atoms if idx not in ref_to_query]
    if missing:
        raise ValueError(
            f"Reference atom indices not found in MCS mapping: {missing}. "
            f"Mapped reference indices: {sorted(ref_to_query)}"
        )

    query_atoms = [ref_to_query[idx] for idx in target_atoms]
    return target_atoms, query_atoms


def mean_distance_from_centroid(df) -> float:
    """Mean Euclidean distance of atoms from the structure centroid (Angstrom)."""
    coords = torch.tensor(df.iloc[2:, 1:4].values.astype("float64"))
    centroid = coords.mean(dim=0)
    return float(torch.sqrt(((coords - centroid) ** 2).sum(dim=1)).mean())


def decade_lr(raw: float) -> float:
    """Snap a positive value to 1, 0.1, 0.01, 0.001, ..."""
    if raw <= 0:
        return 0.01
    return float(10 ** round(math.log10(raw)))


def one_significant_digit(x: float) -> float:
    """Round a positive value to one significant digit (0.331 → 0.3)."""
    if x <= 0:
        return 0.0
    exp = math.floor(math.log10(abs(x)))
    return float(round(x / 10 ** exp) * 10 ** exp)


def proximity_cutoff(df) -> float:
    """coords_proximity threshold: mean |r-center| / 10, one significant digit."""
    return one_significant_digit(mean_distance_from_centroid(df) / 10.0)


def lr_from_structure(df) -> tuple[float, float]:
    """Default lr: mean radius / 100, snapped to a decade. Returns (lr, mean_radius)."""
    radius = mean_distance_from_centroid(df)
    return decade_lr(radius / 100.0), radius


def optimize_numbered(
    target_df,
    query_df,
    target_atoms,
    query_atoms=None,
    init_vars=None,
    lr=None,
    loss_type: str = "mixed",
    sigma: float = 50,
    max_iters: int = 2000,
    debug: bool = False,
):
    
    # prepare the target tensor and the atoms dictionary for one-hot embedding 
    _, _, target, atoms_dictionary = analyze_df(target_df)

    # prepare two seperate vectors for the atoms, and for the corrdinates
    query_coords, query_atom_features, _, _ = analyze_df(query_df, atoms_dictionary)
    
    #initial the variables as zeros
    initial_vars = torch.zeros(6).to(torch.float64)
    
    # if we have calculated initial values, we need to update the initial vars accordingly
    if init_vars != None:
        # reduce COM from query coords
        query_coords -= init_vars [0]

        # reduce COM from target coords
        target[:,:3] -= init_vars [1]
    
        # add to the inital vars
        initial_vars[3] += init_vars[2][0] 
        initial_vars[4] += init_vars[2][1] 
        initial_vars[5] += init_vars[2][2] 

    if lr is None:
        lr, radius = lr_from_structure(target_df)
        if debug:
            print(
                f"[debug] lr={lr:g}  (mean |r-center|={radius:.3f} A / 100 -> decade)",
                flush=True,
            )
    else:
        lr = float(lr)
        if debug:
            print(f"[debug] lr={lr:g}  (user-specified)", flush=True)

    if query_atoms is None:
        query_atoms = target_atoms

    # Restart from the same PCA/COM seed each attempt. If a run hits max_iters
    # without early stopping, double lr and retry (cosine anneal still lr -> lr/10).
    max_lr_retries = 20
    early_stopped = False
    var_tensor = None
    for attempt in range(max_lr_retries + 1):
        lr_min = lr / 10.0
        if debug:
            print(
                f"[debug] Alignment attempt {attempt + 1}/{max_lr_retries + 1}  "
                f"lr={lr:g}  eta_min={lr_min:g}  max_iters={max_iters}  "
                f"loss={loss_type}",
                flush=True,
            )
        start = initial_vars.detach().clone()
        var_tensor = [Variable(start, requires_grad=True)]
        optim = torch.optim.Adam(var_tensor, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, max_iters), eta_min=lr_min,
        )

        output = rotate_and_move(query_coords, var_tensor[0])
        output = torch.concat([output, query_atom_features], dim=1)
        loss = torch.sum(
            loss_function(output[query_atoms], target[target_atoms], sigma, loss_type=loss_type)
        )
        init_loss = float(loss.detach())
        memory_list = [init_loss]
        it = 0
        early_stopped = False
        if debug:
            print(f"[debug]   initial loss={init_loss:.6g}", flush=True)

        while it < max_iters:
            it += 1
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            output = rotate_and_move(query_coords, var_tensor[0])
            output = torch.concat([output, query_atom_features], dim=1)
            loss = torch.sum(
                loss_function(output[query_atoms], target[target_atoms], sigma, loss_type=loss_type)
            )
            loss_value = float(loss.detach())
            memory_list.append(loss_value)
            if len(memory_list) > 500:
                memory_list = memory_list[1:]
            # NaN/Inf never satisfy a numeric plateau check — stop instead of
            # burning through lr doublings on a dead run.
            if not math.isfinite(loss_value):
                early_stopped = True
                if debug:
                    print(
                        f"[debug] Alignment stopped: non-finite loss at iter {it} "
                        f"(lr={lr:g}, initial loss={init_loss:.6g})",
                        flush=True,
                    )
                break
            if it >= 50 and (memory_list[0] - loss_value) <= 1e-6:
                early_stopped = True
                break

        end_loss = memory_list[-1]
        if early_stopped:
            if debug and math.isfinite(end_loss):
                print(
                    f"[debug] Alignment early-stopped with lr={lr:g} "
                    f"(attempt {attempt + 1}, iters={it}, "
                    f"initial loss={init_loss:.6g}, end loss={end_loss:.6g})",
                    flush=True,
                )
            break

        if attempt < max_lr_retries:
            next_lr = lr * 2.0
            if debug:
                print(
                    f"[debug] Alignment hit max_iters={max_iters} with lr={lr:g} "
                    f"(initial loss={init_loss:.6g}, end loss={end_loss:.6g}); "
                    f"retrying with lr={next_lr:g}",
                    flush=True,
                )
            lr = next_lr
        else:
            if debug:
                print(
                    f"[debug] Alignment finished max_iters={max_iters} with lr={lr:g} "
                    f"(no early stop after {max_lr_retries + 1} attempts; "
                    f"initial loss={init_loss:.6g}, end loss={end_loss:.6g})",
                    flush=True,
                )
            break

    #calculate the final coordinates using the variables
    final_coords = rotate_and_move(query_coords, var_tensor[0])

    if init_vars != None:
        # move the final coords according to the target COM
        final_coords += init_vars[1]

    
    # Create the resulted dataframe
    res_df = df_w_symbols(query_df, final_coords)
    
    return res_df


