import torch

def generate_cost(n, low=1, high=10, output_device='cpu', size_threshold=10):
    """
    Efficiently generate an n x n random cost matrix with smart device handling.

    Args:
        n (int): Size of square matrix.
        low (int): Minimum value (inclusive).
        high (int): Maximum value (exclusive).
        output_device (str): 'cpu' or 'cuda' — desired output location.
        size_threshold (int): If output_device is 'cpu', and n^2 > threshold, use GPU for generation.

    Returns:
        torch.Tensor: Cost matrix on the specified output device.
    """

    cuda_available = torch.cuda.is_available()

    if output_device == 'cuda':
        if not cuda_available:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            output_device = 'cpu'  # Fall through to CPU logic
        else:
            return torch.randint(low, high, (n, n), device='cuda')

    # Now output_device is 'cpu' (either originally or via fallback)
    if cuda_available and n > size_threshold:
        # return torch.randint(low, high, (n, n), device='cuda').cpu()
        tensor_gpu = torch.randint(low, high, (n, n), device='cuda')
        tensor_cpu = torch.empty((n, n), dtype=torch.int32, pin_memory=True)
        tensor_cpu.copy_(tensor_gpu, non_blocking=True)
        return tensor_cpu

    return torch.randint(low, high, (n, n), device='cpu')
