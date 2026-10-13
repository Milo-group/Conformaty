import torch


def loss_function(output, target, sigma: float = 20, loss_type: str = "mixed"):
    """
    Compute point-wise loss between output and target.

    Parameters
    ----------
    output : torch.Tensor
        Tensor of shape (N, D) or (D,) with predicted coordinates (+features).
    target : torch.Tensor
        Tensor of the same shape as `output`.
    sigma : float, optional
        Sigma parameter for the SIG loss, by default 20.
    loss_type : {"msd", "mad", "rmsd", "sig", "mixed"}, optional
        Type of loss to compute:
        - "msd"   : mean squared deviation (per point)
        - "mad"   : L2 distance (as used before)
        - "rmsd"  : root mean squared deviation (sqrt of MSD per point)
        - "sig"   : sigmoid-based loss
        - "mixed" : SIG + 5 * MAD (previous default behaviour)

    Returns
    -------
    torch.Tensor
        Loss tensor per point.
    """

    diff = target - output
    sq = torch.sum(torch.pow(diff, 2), -1)  # ||r_i||^2
    MAD = torch.mean(torch.abs(diff), -1)
    MSD = torch.mean(sq)
    RMSD = torch.sqrt(torch.clamp(MSD, min=1e-24))
    SIG = 1 - 2 / (1 + torch.exp(sigma * MAD))  # 20,30

    if loss_type.lower() == "msd":
        return MSD
    elif loss_type.lower() == "mad":
        return MAD
    elif loss_type.lower( ) == "rmsd":
        return RMSD
    elif loss_type.lower() == "sig":
        return SIG
    else:
        # Default / "mixed": original behaviour SIG + 5*MAD
        return SIG + 5 * MAD


def loss_calc(output, target, sigma: float = 20, loss_type: str = "mixed"):

    loss = 0

    for point in output:
        loss += torch.min(loss_function(point, target, sigma=sigma, loss_type=loss_type))

    return loss
