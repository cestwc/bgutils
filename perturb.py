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

    return perturbed_C


import torch

def perturb_l0(C, X, ratio=0.1, num_perturb=None, large_value=1e9, size_threshold=10):
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

    n, m = C.shape
    orig_device = C.device
    if torch.cuda.is_available() and n > size_threshold:
        device = torch.device('cuda')
    else:
        device = orig_device

    # Move data only if needed
    if device != orig_device:
        # candidate_indices = candidate_indices.to(device)
        X = X.to(device)
        C_work = C.to(device)
    else:
        C_work = C

    candidate_indices = torch.nonzero(X, as_tuple=False)


    num_candidates = candidate_indices.size(0)
    if num_perturb is None:
        num_perturb = max(1, int(ratio * num_candidates))

    # Randomly select indices to perturb
    selected = candidate_indices[torch.randperm(num_candidates, device=device)[:num_perturb]]

    # Apply perturbation
    perturbed_C = C_work.clone()
    perturbed_C[selected[:, 0], selected[:, 1]] = large_value

    # Return to original device if needed
    if device == orig_device:
        return perturbed_C
    else:
        tensor_cpu = torch.empty((n, n), dtype=torch.int32, pin_memory=True)
        tensor_cpu.copy_(perturbed_C, non_blocking=True)
        return tensor_cpu

