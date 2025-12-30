"""
CPU SERIAL IMPLEMENTATION
-------------------------
This module implements brightness and contrast enhancement
using explicit nested loops. It represents the baseline,
non-optimized CPU approach.
"""

import numpy as np


def brightness_cpu(image: np.ndarray, value: int) -> np.ndarray:
    """
    Adjust image brightness using a serial CPU approach.

    Formula:
        new_pixel = pixel + value

    Pixel values are clipped to [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or RGB image (uint8).
    value : int
        Brightness offset (positive or negative).

    Returns
    -------
    np.ndarray
        Brightness-adjusted image.
    """
    output = image.copy()
    height, width = image.shape[:2]

    for i in range(height):
        for j in range(width):
            if image.ndim == 3:  # RGB image
                for c in range(3):
                    px = image[i, j, c] + value
                    output[i, j, c] = min(255, max(0, px))
            else:  # Grayscale image
                px = image[i, j] + value
                output[i, j] = min(255, max(0, px))

    return output


def contrast_cpu(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image contrast using a serial CPU approach.

    Formula:
        new_pixel = 128 + factor * (pixel - 128)

    Pixel values are clipped to [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Input grayscale or RGB image (uint8).
    factor : float
        Contrast scaling factor (>1 increases contrast).

    Returns
    -------
    np.ndarray
        Contrast-adjusted image.
    """
    output = image.copy()
    height, width = image.shape[:2]

    for i in range(height):
        for j in range(width):
            if image.ndim == 3:
                for c in range(3):
                    px = 128 + factor * (image[i, j, c] - 128)
                    output[i, j, c] = min(255, max(0, px))
            else:
                px = 128 + factor * (image[i, j] - 128)
                output[i, j] = min(255, max(0, px))

    return output
