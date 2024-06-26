from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import numpy as np
import logging


# Global variables 

logger = logging.getLogger(__name__)
float_tolerance = 1e-7


# Main function 

def generate_base_image(image, sigma, assumed_blur):
    """
    Generate a base image from the input image by upsampling it by 2 in both directions and applying blurring.

    Args:
        image (numpy.ndarray): Input image (2D array).
        sigma (float): Standard deviation parameter passed to the blurring function.
        assumed_blur (float): Blurring ratio.

    Returns:
        numpy.ndarray: Blurred and upsampled image.
    """
    # resize the image by factor of 2 to cover a range of scales
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    
    # calculating the standard deviation (sigma_diff) for Gaussian blur
    # sqrt(max(..., 0.01)) ensures that sigma_diff is always at least 0.1 to avoid division by zero in the subsequent Gaussian blur operation.
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

def compute_number_of_octaves(image_shape):
    """
    Compute the number of octaves in an image pyramid based on the input image shape.

    Args:
        image_shape (tuple): A tuple representing the dimensions (width and height) of the input image.

    Returns:
        int: Number of octaves.
    """
    # In scale-space pyramid construction, the number of octaves is typically determined by the smallest dimension of the image
    return int(round(log(min(image_shape)) / log(2) - 1))

def generate_gaussian_kernels(sigma, num_intervals):
    """
    Generates a list of Gaussian kernels at which to blur the input image.
    The default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.

    Args:
        sigma (float): Standard deviation parameter for Gaussian blurring.
        num_intervals (int): Number of intervals (scales) in the image pyramid.

    Returns:
        numpy.ndarray: Array of Gaussian kernel values corresponding to different scales.
    """
    num_images_per_octave = num_intervals + 3
    
    # The scaling factor used to compute subsequent sigma values
    k = 2 ** (1. / num_intervals)
    
    # An array to store the computed Gaussian kernel values
    # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels = zeros(num_images_per_octave)  
    
    # The first element of the array is set to the input sigma
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
        
    return gaussian_kernels

def generate_gaussian_images(image, num_octaves, gaussian_kernels):
    """
    Generates a scale-space pyramid of Gaussian images.

    Args:
        image (numpy.ndarray): Input image (2D array).
        num_octaves (int): Number of octaves in the image pyramid.
        gaussian_kernels (numpy.ndarray): Array of Gaussian kernel values corresponding to different scales.

    Returns:
        numpy.ndarray: Array of arrays containing Gaussian images for each octave.
    """
    
    # Initializing an empty list to store the Gaussian images for each octave
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        
        # first image in octave already has the correct blur
        gaussian_images_in_octave.append(image) 
         
        for gaussian_kernel in gaussian_kernels[1:]:
            # Apply Gaussian blur to the current image using the specified kernel
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
            
        # Append the entire list of Gaussian images for the current octave
        gaussian_images.append(gaussian_images_in_octave)
        
        # Update the image for the next octave by resizing the base image from the current octave
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
        
    return array(gaussian_images,dtype=object)

def generate_DoG_images(gaussian_images):
    """
    Generates a Difference-of-Gaussians (DoG) image pyramid.

    Args:
        gaussian_images (numpy.ndarray): Array of arrays containing Gaussian images for each octave.

    Returns:
        numpy.ndarray: Array of arrays containing DoG images for each octave.
    """
    # Initialize an empty list to store DoG images
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            # Compute the absolute difference between corresponding pixel values
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
            
        # Append the entire list of DoG images for the current octave
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)

def find_scale_space_extrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """
    Finds pixel positions of all scale-space extrema in the image pyramid.

    Args:
        gaussian_images (numpy.ndarray): Array of arrays containing Gaussian images for each octave.
        dog_images (numpy.ndarray): Array of arrays containing DoG images for each octave.
        num_intervals (int): Number of intervals (scales) in the image pyramid.
        sigma (float): Standard deviation parameter for Gaussian blurring.
        image_border_width (int): Width of the border to exclude near the image edges.
        contrast_threshold (float, optional): Threshold for detecting significant extrema. Defaults to 0.04.

    Returns:
        list: List of keypoints (with orientations) found across different scales and octaves.
    """
    
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    
    # Initialize an empty list to store keypoints
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            # Iterate over pixel positions within the specified border width
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if is_pixel_an_extremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localize_extremum_via_quadraticfit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = compute_keypoints_with_orientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints

def is_pixel_an_extremum(first_subimage, second_subimage, third_subimage, threshold):
    """
    Returns True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors,
    False otherwise.

    Args:
        first_subimage (numpy.ndarray): 3x3 array of pixel values from the previous scale.
        second_subimage (numpy.ndarray): 3x3 array of pixel values from the current scale (center element).
        third_subimage (numpy.ndarray): 3x3 array of pixel values from the next scale.
        threshold (int): Threshold for detecting significant extrema.

    Returns:
        bool: True if the center pixel is an extremum, False otherwise.
    """
    center_pixel_value = second_subimage[1, 1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return all(center_pixel_value >= first_subimage) and \
                   all(center_pixel_value >= third_subimage) and \
                   all(center_pixel_value >= second_subimage[0, :]) and \
                   all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= first_subimage) and \
                   all(center_pixel_value <= third_subimage) and \
                   all(center_pixel_value <= second_subimage[0, :]) and \
                   all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False

def localize_extremum_via_quadraticfit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """
    Iteratively refines pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors.

    Args:
        i (int): Row index of the initial extremum position.
        j (int): Column index of the initial extremum position.
        image_index (int): Index of the current DoG image in the octave.
        octave_index (int): Index of the current octave.
        num_intervals (int): Number of intervals (scales) in the image pyramid.
        dog_images_in_octave (numpy.ndarray): Array of DoG images for the current octave.
        sigma (float): Standard deviation parameter for Gaussian blurring.
        contrast_threshold (float): Threshold for detecting significant extrema.
        image_border_width (int): Width of the border to exclude near the image edges.
        eigenvalue_ratio (float, optional): Ratio of eigenvalues for stability. Defaults to 10.
        num_attempts_until_convergence (int, optional): Maximum number of attempts for convergence. Defaults to 5.

    Returns:
        tuple: A tuple containing the localized keypoint (i, j, image_index) and the localized image index.
            If the extremum moves outside the image, returns None.
    """
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = compute_gradient_at_center_pixel(pixel_cube)
        hessian = compute_hessian_at_center_pixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        logger.debug('Updated extremum moved outside of image before reaching convergence. Skipping...')
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        logger.debug('Exceeded maximum number of attempts without reaching convergence for this extremum. Skipping...')
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

def compute_gradient_at_center_pixel(pixel_array):
    """
    Approximates the gradient at the center pixel [1, 1, 1] of a 3x3x3 array using the central difference formula of order O(h^2),
    where h is the step size.

    Args:
        pixel_array (numpy.ndarray): A 3D array representing pixel values in a 3x3x3 neighborhood around a central pixel.

    Returns:
        numpy.ndarray: An array containing the gradient components [dx, dy, ds], where:
            - dx: Gradient in the x-direction (horizontal)
            - dy: Gradient in the y-direction (vertical)
            - ds: Gradient in the s-direction (scale or depth)
    """
   
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])

def compute_hessian_at_center_pixel(pixel_array):
    """
    Approximates the Hessian matrix at the center pixel [1, 1, 1] of a 3x3x3 array using the central difference formula of order O(h^2),
    where h is the step size.

    Args:
        pixel_array (numpy.ndarray): A 3D array representing pixel values in a 3x3x3 neighborhood around a central pixel.

    Returns:
        numpy.ndarray: A 3x3 matrix containing the second derivatives in the x, y, and s directions. The matrix elements are as follows:
            - dxx: Second derivative with respect to x (horizontal)
            - dyy: Second derivative with respect to y (vertical)
            - dss: Second derivative with respect to s (scale or depth)
            - dxy: Mixed derivative with respect to x and y
            - dxs: Mixed derivative with respect to x and s
            - dys: Mixed derivative with respect to y and s
    """
   
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    # Calculate second derivatives
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]   # Second derivative in the x-direction
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]   # Second derivative in the y-direction
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]   # Second derivative in the s-direction
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])   # Mixed derivative with respect to x and y
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])   # Mixed derivative with respect to x and s
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])   # Mixed derivative with respect to y and s
    
    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])
    
def compute_keypoints_with_orientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """
    Computes orientations for each keypoint based on gradient information in the neighborhood.

    Args:
        keypoint (cv2.KeyPoint): A keypoint object representing a detected feature point.
        octave_index (int): The index of the octave in which the keypoint was detected.
        gaussian_image (numpy.ndarray): The Gaussian-blurred image pyramid.
        radius_factor (float, optional): A scaling factor for the neighborhood radius around the keypoint (default is 3).
        num_bins (int, optional): The number of bins in the orientation histogram (default is 36).
        peak_ratio (float, optional): A threshold for identifying peaks in the histogram (default is 0.8).
        scale_factor (float, optional): A scaling factor for the keypoint size (default is 1.5).

    Returns:
        list: A list of updated keypoints with orientations.
    """
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

def compare_keypoints(keypoint1, keypoint2):
    """
    Compares two keypoints and determines if keypoint1 is "less" than keypoint2 based on various attributes.

    Args:
        keypoint1 (cv2.KeyPoint): The first keypoint object.
        keypoint2 (cv2.KeyPoint): The second keypoint object.

    Returns:
        int: A negative value if keypoint1 is less than keypoint2, zero if they are equal, and a positive value otherwise.
    """
    
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def remove_duplicate_keypoints(keypoints):
    """
    Sorts keypoints and removes duplicate keypoints based on their attributes.

    Args:
        keypoints (list): A list of cv2.KeyPoint objects representing detected feature points.

    Returns:
        list: A list of unique keypoints after removing duplicates.
    """
    # Sort keypoints based on custom comparison function
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compare_keypoints))
    unique_keypoints = [keypoints[0]]

    # Iterate through keypoints and keep only unique ones
    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

def convert_keypoints_to_input_image_size(keypoints):
    """
    Converts keypoints' point, size, and octave to the input image size.

    Args:
        keypoints (list): A list of cv2.KeyPoint objects representing detected feature points.

    Returns:
        list: A list of keypoints with updated attributes (coordinates, size, and octave) based on the input image size.
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

def unpack_octave(keypoint):
    """
    Computes octave, layer, and scale from a keypoint's octave attribute.

    Args:
        keypoint (cv2.KeyPoint): A keypoint object.

    Returns:
        tuple: A tuple containing the computed octave, layer, and scale values.
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale

def generate_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """
    Generates descriptors for each keypoint based on gradient information within a descriptor window.

    Args:
        keypoints (list): A list of cv2.KeyPoint objects representing detected feature points.
        gaussian_images (numpy.ndarray): The Gaussian-blurred image pyramid.
        window_width (int, optional): Width of the descriptor window (default is 4).
        num_bins (int, optional): Number of bins in the orientation histogram (default is 8).
        scale_multiplier (float, optional): Scale multiplier for descriptor size (default is 3).
        descriptor_max_value (float, optional): Maximum value for descriptor elements (default is 0.2).

    Returns:
        numpy.ndarray: An array of descriptors corresponding to the input keypoints.
    """
   
    logger.debug('Generating descriptors...')
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpack_octave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')



def sift(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    image = image.astype('float32')
    base_image = generate_base_image(image, sigma, assumed_blur)
    num_octaves = compute_number_of_octaves(base_image.shape)
    gaussian_kernels = generate_gaussian_kernels(sigma, num_intervals)
    gaussian_images = generate_gaussian_images(base_image, num_octaves, gaussian_kernels)
    dog_images = generate_DoG_images(gaussian_images)
    keypoints = find_scale_space_extrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = remove_duplicate_keypoints(keypoints)
    keypoints = convert_keypoints_to_input_image_size(keypoints)
    descriptors = generate_descriptors(keypoints, gaussian_images)
    return keypoints, descriptors