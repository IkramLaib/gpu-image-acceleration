"""
GPU CUDA IMPLEMENTATION
----------------------
CUDA kernels for brightness and contrast enhancement
using Numba.
"""

from numba import cuda


@cuda.jit
def brightness_kernel(img, out, value):
    """
    CUDA kernel for brightness adjustment.

    Each thread processes one pixel location.
    """
    x, y = cuda.grid(2)

    if x < img.shape[0] and y < img.shape[1]:
        if img.ndim == 3:
            for c in range(3):
                px = img[x, y, c] + value
                out[x, y, c] = min(255, max(0, px))
        else:
            px = img[x, y] + value
            out[x, y] = min(255, max(0, px))


@cuda.jit
def contrast_kernel(img, out, factor):
    """
    CUDA kernel for contrast adjustment.

    Each thread applies the contrast formula
    to one pixel location.
    """
    x, y = cuda.grid(2)

    if x < img.shape[0] and y < img.shape[1]:
        if img.ndim == 3:
            for c in range(3):
                px = 128 + factor * (img[x, y, c] - 128)
                out[x, y, c] = min(255, max(0, px))
        else:
            px = 128 + factor * (img[x, y] - 128)
            out[x, y] = min(255, max(0, px))
