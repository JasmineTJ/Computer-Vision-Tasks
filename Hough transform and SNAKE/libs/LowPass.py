#
# Low Pass libs Implementations
#

import numpy as np
import scipy.stats as st
from scipy.signal import convolve2d
from typing import Union

def create_square_kernel(size: int, mode: str, sigma: Union[int, float] = None) -> np.ndarray:
    """
        Create/Calculate a square kernel for different low pass filter modes

    :param size: Kernel Size
    :param mode: Low Pass Filter Mode ['ones' -> Average Filter Mode, 'gaussian', 'median' ]
    :param sigma: Variance amount in case of 'Gaussian' mode
    :return: Square Array Kernel
    """
    if mode == 'ones':
        return np.ones((size, size))
    elif mode == 'gaussian':
        space = np.linspace(np.sqrt(sigma), -np.sqrt(sigma), size * size)
        kernel1d = np.diff(st.norm.cdf(space))
        kernel2d = np.outer(kernel1d, kernel1d)
        return kernel2d / kernel2d.sum()


def apply_kernel(source: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """
        Calculate/Apply Convolution of two arrays, one being the kernel
        and the other is the image

    :param source: First Array
    :param kernel: Calculated Kernel
    :param mode: Convolution mode ['valid', 'same']
    :return: Convoluted Result
    """
    src = np.copy(source)

    # Check for Grayscale Image
    if len(src.shape) == 2 or src.shape[-1] == 1:
        conv = convolve2d(src, kernel, mode)
        return conv.astype('uint8')

    out = []
    # Apply Kernel using Convolution
    for channel in range(src.shape[-1]):
        conv = convolve2d(src[:, :, channel], kernel, mode)
        out.append(conv)
    return np.stack(out, -1)

def gaussian_filter(source: np.ndarray, shape: int = 5, sigma: Union[int, float] = 64) -> np.ndarray:
    """
        Gaussian Low Pass Filter Implementation
    :param source: Image to Apply Filter to
    :param shape: An Integer that denotes th Kernel size if 3
                  then the kernel is (3, 3)
    :param sigma: Standard Deviation
    :return: Filtered Image
    """
    src = np.copy(source)

    # Create a Gaussian Kernel
    kernel = create_square_kernel(shape, 'gaussian', sigma)

    # Apply the Kernel
    out = apply_kernel(src, kernel, 'same')
    return out.astype('uint8')