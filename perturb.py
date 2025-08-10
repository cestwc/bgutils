import torch

def perturb_sparse(C, ratio=0.05, max_change=3, size_threshold=100):
    """
    Perturb a sparse-like matrix C by randomly modifying a fraction of its elements.
    Automatically uses GPU if:
      - C has more than `size_threshold` elements
      - CUDA is available

    Args:
        C (torch.Tensor): Input 2D tensor.
        ratio (float): Fraction of elements to perturb.
        max_change (int): Maximum positive/negative perturbation.
        size_threshold (int): Number of elements above which GPU is used if available.

    Returns:
        torch.Tensor: Perturbed tensor on the same device as input.
    """
    assert C.dim() == 2, "Input tensor must be 2D"

    n, m = C.shape
    num_elements = n * m
    num_perturb = max(1, int(ratio * num_elements))

    # Decide device
    original_device = C.device
    use_gpu = (num_elements > size_threshold) and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else original_device

    # Move to target device for processing
    C_work = C.to(device)

    # Vectorized perturbation
    flat_indices = torch.randperm(num_elements, device=device)[:num_perturb]
    rows = flat_indices // m
    cols = flat_indices % m
    deltas = torch.randint(-max_change, max_change + 1, size=(num_perturb,), device=device)

    perturbed_C = C_work.clone()
    perturbed_C[rows, cols] += deltas
    perturbed_C[rows, cols] = torch.clamp(perturbed_C[rows, cols], min=0)

    # Move back to original device if needed
    if perturbed_C.device != original_device:
        tensor_cpu = torch.empty((n, n), dtype=torch.int32, pin_memory=True)
        tensor_cpu.copy_(perturbed_C, non_blocking=True)
        return tensor_cpu

    return perturbed_C.clone().contiguous()


import torch

def perturb_l0(C, X, ratio=None, num_perturb=None, large_value=1e9, size_threshold=10):
    """
    Efficiently perturb matrix C at positions where X == 1 by setting selected entries to a large value.

    Args:
        C (torch.Tensor): Cost matrix (n x m).
        X (torch.Tensor): Binary mask (same shape as C).
        ratio (float): Fraction of positions where X == 1 to perturb. Ignored if num_perturb is specified.
        num_perturb (int or None): Number of positions to perturb. Overrides ratio if provided.
        large_value (float): Value to assign to selected positions.
        size_threshold (int): Threshold above which GPU is used temporarily if CUDA is available.

    Returns:
        torch.Tensor: Perturbed matrix (same device as C).
    """

    assert (ratio is not None) ^ (num_perturb is not None)

    n, m = C.shape

    perturbed_C = C.clone()
    # X_work = X.clone()
        

    candidate_indices = torch.nonzero(X, as_tuple=False)

    num_candidates = candidate_indices.size(0)

    if ratio is not None:
        num_perturb = int(ratio * num_candidates)

    # Randomly select indices to perturb
    selected = candidate_indices[torch.randperm(num_candidates, device=C.device)[:num_perturb]]

    # Apply perturbation
    # perturbed_C = C_work.clone()
    perturbed_C[selected[:, 0], selected[:, 1]] = large_value
    return perturbed_C

import torch

def perturb_l0_force_assign(C, X, ratio=None, num_perturb=None, large_value=1e9):
    """
    For selected 1-entries in X, make all other entries in their rows/columns large,
    leaving the selected cells untouched—biasing the solver to pick those cells.
    """
    assert (ratio is not None) ^ (num_perturb is not None), "Provide exactly one of ratio or num_perturb."

    n, m = C.shape
    perturbed_C = C.clone()

    # candidate 1-positions
    candidate_indices = torch.nonzero(X, as_tuple=False)
    num_candidates = candidate_indices.size(0)
    if num_candidates == 0:
        return perturbed_C

    if ratio is not None:
        num_perturb = max(0, min(num_candidates, int(ratio * num_candidates)))

    if num_perturb == 0:
        return perturbed_C

    # pick which (i, j) to force-assign
    perm = torch.randperm(num_candidates, device=C.device)
    selected = candidate_indices[perm[:num_perturb]]  # shape [k, 2]

    rows = selected[:, 0]
    cols = selected[:, 1]

    # Build row/column masks
    row_mask = torch.zeros(n, dtype=torch.bool, device=C.device)
    row_mask[rows] = True
    col_mask = torch.zeros(m, dtype=torch.bool, device=C.device)
    col_mask[cols] = True

    # All cells in selected rows or columns...
    big_mask = row_mask[:, None] | col_mask[None, :]

    # ...except the selected cells themselves
    exclude = torch.zeros_like(C, dtype=torch.bool)
    exclude[rows, cols] = True
    final_mask = big_mask & ~exclude

    # Apply large values
    perturbed_C[final_mask] = large_value

    # (Optional) really lock it in by making the selected cells extra small:
    # perturbed_C[rows, cols] = perturbed_C[rows, cols] - large_value

    return perturbed_C
