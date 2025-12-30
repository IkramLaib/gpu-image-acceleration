"""
UTILITY FUNCTIONS
-----------------
GPU execution helper functions.
"""

import time
from numba import cuda


def run_gpu(kernel, image, parameter):
    """
    Execute a CUDA kernel and measure execution time.

    Parameters
    ----------
    kernel : cuda.jit function
        CUDA kernel to execute.
    image : np.ndarray
        Input image.
    parameter : int or float
        Kernel parameter (brightness or contrast).

    Returns
    -------
    output : np.ndarray
        Result copied back from GPU.
    elapsed : float
        Kernel execution time in seconds.
    """
    d_img = cuda.to_device(image)
    d_out = cuda.device_array_like(image)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1],
    )

    start = time.time()
    kernel[blocks_per_grid, threads_per_block](d_img, d_out, parameter)
    cuda.synchronize()
    elapsed = time.time() - start

    return d_out.copy_to_host(), elapsed
