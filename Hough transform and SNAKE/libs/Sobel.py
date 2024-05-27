import cv2
import numpy as np
from scipy.signal import convolve2d


def apply_kernel(source: np.ndarray, horizontal_kernel: np.ndarray, vertical_kernel: np.ndarray,
                 ReturnEdge: bool = False):
    """
        Convert image to gray scale and convolve with kernels
        :param source: Image to apply kernel to
        :param horizontal_kernel: The horizontal array of the kernel
        :param vertical_kernel: The vertical array of the kernel
        :param ReturnEdge: Return Horizontal & Vertical Edges
        :return: The result of convolution
    """
    # convert to gray scale if not already
    if len(source.shape) > 2:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    else:
        gray = source

    # convolution
    horizontal_edge = convolve2d(gray, horizontal_kernel)
    vertical_edge = convolve2d(gray, vertical_kernel)

    mag = np.sqrt(pow(horizontal_edge, 2.0) + pow(vertical_edge, 2.0))
    if ReturnEdge:
        return mag, horizontal_edge, vertical_edge
    return mag

def sobel_edge(source: np.ndarray, GetMagnitude: bool = True, GetDirection: bool = False):
    """
        Apply Sobel Operator to detect edges
        :param source: Image to detect edges in
        :param GetMagnitude: Get Magnitude of horizontal and vertical edges
        :param GetDirection: Get Gradient direction in Pi Terms
        :return: edges image
    """

    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.flip(horizontal.T)
    mag, HorizontalEdge, VerticalEdge = apply_kernel(source, horizontal, vertical, True)

    HorizontalEdge = HorizontalEdge[:-2, :-2]
    VerticalEdge = VerticalEdge[:-2, :-2]

    if GetMagnitude == False:
        return HorizontalEdge, VerticalEdge

    if GetDirection:
        Direction = np.arctan2(VerticalEdge, HorizontalEdge)
        return mag, Direction

    return mag
