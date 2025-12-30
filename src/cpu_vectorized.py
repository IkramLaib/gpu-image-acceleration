"""
CPU VECTORIZED IMPLEMENTATION
-----------------------------
This module implements brightness and contrast enhancement
using NumPy vectorized operations.
"""


import numpy as np


def brightness_vectorized(image: np.ndarray, value: int) -> np.ndarray:
    """
    Vectorized brightness adjustment.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    value : int
        Brightness offset.

    Returns
    -------
    np.ndarray
        Brightness-adjusted image.
    """
    return np.clip(image + value, 0, 255).astype(np.uint8)


def contrast_vectorized(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Vectorized contrast adjustment.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8).
    factor : float
        Contrast scaling factor.

    Returns
    -------
    np.ndarray
        Contrast-adjusted image.
    """
    return np.clip(128 + factor * (image - 128), 0, 255).astype(np.uint8)
