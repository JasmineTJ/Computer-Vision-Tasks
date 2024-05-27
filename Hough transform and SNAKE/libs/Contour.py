import itertools
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from libs.Sobel import sobel_edge
from libs.LowPass import gaussian_filter


def iterate_contour(image: np.ndarray, contour_x: np.ndarray, contour_y: np.ndarray,
                    external_energy: np.ndarray, window_coordinates: list,
                    alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function iterates over a contour and optimizes its position based on internal and external energies.

    Parameters:
    image (np.ndarray): The source image.
    contour_x (np.ndarray): The x-coordinates of the initial contour.
    contour_y (np.ndarray): The y-coordinates of the initial contour.
    external_energy (np.ndarray): The external energy array (e.g., gradient, edge map) of the same size as the image.
    window_coordinates (list): The list of coordinates in the window around each contour point to be considered for the new position.
    alpha (float): The weight of the continuity (first-order) term in the internal energy.
    beta (float): The weight of the curvature (second-order) term in the internal energy.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The optimized x and y coordinates of the contour.

    Note:
    The function uses a greedy algorithm to minimize the energy function. For each point on the contour, it considers moving the point to each position in a window around the original position and computes the energy at each new position. The point is then moved to the position with the lowest energy.
    """

    # Copy the image and contour coordinates
    image_copy = np.copy(image)
    contour_x_copy = np.copy(contour_x)
    contour_y_copy = np.copy(contour_y)

    num_points = len(contour_x_copy)
    
    # Iterate over each point in the contour
    for point_index in range(num_points):
        min_energy = np.inf
        new_x = None
        new_y = None
        # Iterate over each coordinate in the window
        for window_coord in window_coordinates:
            # Create temporary contours with point shifted to a coordinate
            temp_x, temp_y = np.copy(contour_x_copy), np.copy(contour_y_copy)
            temp_x[point_index] = min(temp_x[point_index] + window_coord[0], image_copy.shape[1] - 1)
            temp_y[point_index] = min(temp_y[point_index] + window_coord[1], image_copy.shape[0] - 1)

            # Calculate energy at the new point
            try:
                total_energy = - external_energy[temp_y[point_index], temp_x[point_index]] + calculate_internal_energy(temp_x, temp_y, alpha, beta)
            except:
                pass

            # Save the point if it has the lowest energy in the window
            if total_energy < min_energy:
                min_energy = total_energy
                new_x = temp_x[point_index]
                new_y = temp_y[point_index]

        # Shift the point in the contour to its new location with the lowest energy
        contour_x_copy[point_index] = new_x
        contour_y_copy[point_index] = new_y

    return contour_x_copy, contour_y_copy

def create_square_contour(image: np.ndarray, num_x_points: int, num_y_points: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    This function creates a square contour and a neighborhood window around each point on the contour.

    Parameters:
    image (np.ndarray): The source image.
    num_x_points (int): The number of points along the x-axis of the contour.
    num_y_points (int): The number of points along the y-axis of the contour.

    Returns:
    Tuple[np.ndarray, np.ndarray, list]: The x and y coordinates of the contour and the coordinates of the neighborhood window.

    Note:
    The function generates a square contour by creating four lists of points for the top, right, bottom, and left sides of the square. It then concatenates these lists into a single array for the x and y coordinates of the contour. The contour is then shifted to a specific location in the image. Finally, the function generates a neighborhood window around each point on the contour.
    """

    step = 5

    # Create x points lists for each side of the square
    top_x = np.arange(0, num_x_points, step)
    right_x = np.repeat((num_x_points) - step, num_x_points // step)
    bottom_x = np.flip(top_x)
    left_x = np.repeat(0, num_x_points // step)

    # Create y points list for each side of the square
    top_y = np.repeat(0, num_y_points // step)
    right_y = np.arange(0, num_y_points, step)
    bottom_y = np.repeat(num_y_points - step, num_y_points // step)
    left_y = np.flip(right_y)

    # Concatenate all the lists in one array for x and y coordinates of the contour
    contour_x = np.array([top_x, right_x, bottom_x, left_x]).ravel()
    contour_y = np.array([top_y, right_y, bottom_y, left_y]).ravel()

    # Shift the shape to a specific location in the image
    contour_x = contour_x + (image.shape[1] // 2) - 95
    contour_y = contour_y + (image.shape[0] // 2) - 40

    # Create neighborhood window
    window_coordinates = generate_window_coordinates(5)

    return contour_x, contour_y, window_coordinates

def create_ellipse_contour(image: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    This function creates an ellipse contour and a neighborhood window around each point on the contour.

    Parameters:
    image (np.ndarray): The source image.
    num_points (int): The number of points along the contour.

    Returns:
    Tuple[np.ndarray, np.ndarray, list]: The x and y coordinates of the contour and the coordinates of the neighborhood window.

    Note:
    The function generates an ellipse contour by creating two lists of points for the x and y coordinates of the contour. The contour is then shifted to a specific location in the image. Finally, the function generates a neighborhood window around each point on the contour.
    """

    # Create x and y lists coordinates to initialize the contour
    t = np.arange(0, num_points / 10, 0.1)

    # Coordinates for Circles_v2.png image
    contour_x = (image.shape[1] // 2) + 117 * np.cos(t) - 100
    contour_y = (image.shape[0] // 2) + 117 * np.sin(t) + 50

    # Convert to integer type
    contour_x = contour_x.astype(int)
    contour_y = contour_y.astype(int)

    # Create neighborhood window
    window_coordinates = generate_window_coordinates(5)

    return contour_x, contour_y, window_coordinates

def generate_window_coordinates(size: int):
    """
    Generates a list of all possible coordinates inside a window of a given size.

    Parameters:
    size (int): The size of the window.

    Returns:
    list: The list of all possible coordinates inside the window.
    """
    # Generate a list of all possible point values based on the window size
    points = list(range(-size // 2 + 1, size // 2 + 1))

    # Generate all possible coordinates inside the window using Cartesian product
    coordinates = list(itertools.product(points, repeat=2))
    return coordinates

def calculate_internal_energy(contour_x, contour_y, alpha: float, beta: float):
    """
    Calculates the internal energy of a contour.

    Parameters:
    contour_x (np.ndarray): The x-coordinates of the contour.
    contour_y (np.ndarray): The y-coordinates of the contour.
    alpha (float): The weight of the continuity (first-order) term in the internal energy.
    beta (float): The weight of the curvature (second-order) term in the internal energy.

    Returns:
    float: The internal energy of the contour.
    """
    # Join the x and y coordinates and transpose the array
    points = np.array((contour_x, contour_y)).T

    # Calculate the continuous energy
    prev_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    displacements = points - prev_points
    point_distances = np.linalg.norm(displacements, axis=1)
    mean_distance = np.mean(point_distances)
    continuous_energy = np.sum((point_distances - mean_distance) ** 2)

    # Calculate the curvature energy
    curvature_separated = prev_points - 2 * points + next_points
    curvature_energy = np.sum(np.linalg.norm(curvature_separated, axis=1) ** 2)

    return alpha * continuous_energy + beta * curvature_energy

def calculate_external_energy(image, w_line, w_edge):
    """
    Calculates the external energy of an image.

    Parameters:
    image (np.ndarray): The source image.
    w_line (float): The weight of the line energy.
    w_edge (float): The weight of the edge energy.

    Returns:
    np.ndarray: The external energy of the image.
    """
    # Convert to gray scale if the image is not already in gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) > 2 else image

    # Apply Gaussian Filter to smooth the image
    e_line = gaussian_filter(gray, 7, 7 * 7)

    # Get Gradient Magnitude & Direction
    e_edge, _ = sobel_edge(e_line, GetDirection=True)

    return w_line * e_line + w_edge * e_edge[1:-1, 1:-1]