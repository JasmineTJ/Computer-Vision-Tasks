import numpy
import numpy as np
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage
# from PyQt5.QtWidgets import QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QLayout
from PyQt5.QtWidgets import *
import sys
import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve
import time
import copy
from scipy import signal
import cv2
import apply_sift

import scipy

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import time
import scipy.ndimage as ndimage
from scipy.signal import convolve2d

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=20, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.tabWidget_2 = self.findChild(QTabWidget, 'tabWidget_2')
        # Define noisy_image as an instance variable
        self.noisy_image = None
        # Load the UI Page
        uic.loadUi(r'GUI.ui', self)
        # ui setup
        self.handle_ui()
        self.radioButton_harris.setChecked(True)
        self.img_ncc_button.clicked.connect(self.browse_input_image)
        self.target_ncc_button.clicked.connect(self.browse_target_image)
        self.pushButton_4.clicked.connect(self.browse_input_image)
        self.Apply_Harris.clicked.connect(self.handle_harris)
        self.threshold_slider.valueChanged.connect(self.slider_value_changed)
        self.upload1_sift_Button.clicked.connect(self.browse_input_image)
        self.upload2_sift_Button.clicked.connect(lambda: self.browse_image(self.sift2_graphicsView, self.sift2_scene))
        self.sift_apply_button.clicked.connect(self.apply_sift)

    def handle_ui(self):
        # Create a QGraphicsScene
        self.input_scene = QtWidgets.QGraphicsScene()
        self.input_img_ncc.setScene(self.input_scene)
        self.input_scene2 = QtWidgets.QGraphicsScene()
        self.input_target_ncc.setScene(self.input_scene2)
        self.output_scene = QtWidgets.QGraphicsScene()
        self.out_ncc.setScene(self.output_scene)
        self.input_harris_scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.input_harris_scene)
        self.output_harris_scene = QtWidgets.QGraphicsScene()
        self.graphicsView_2.setScene(self.output_harris_scene)
        
        self.sift1_scene = QtWidgets.QGraphicsScene()
        self.sift1_graphicsView.setScene(self.sift1_scene)
        
        self.sift2_scene = QtWidgets.QGraphicsScene()
        self.sift2_graphicsView.setScene(self.sift2_scene)
        
        self.sift_res_scene = QtWidgets.QGraphicsScene()
        self.sift_res_graphicsView.setScene(self.sift_res_scene)

        ####################################################################################################
        self.harris_threshold=0
        # # Create a QGraphicsScene in the handle_ui function
        # self.input_scene_harris = QtWidgets.QGraphicsScene()
        # self.graphicsView.setScene(self.input_scene_harris)
        #
        # self.output_scene_harris = QtWidgets.QGraphicsScene()
        # self.graphicsView_2.setScene(self.output_scene_harris)
        self.sift = cv2.SIFT_create()

    def handle_harris(self):
        if self.radioButton_harris.isChecked():
            self.apply_harris()
        else:
            self.lambda_minus_corner_detection()

    def slider_value_changed(self, value):
        print(f"Slider value changed: {value}")
        self.harris_threshold = value/100
        val = str(value/100)
        self.label_23.setText(f"{str(value/100)}")

        # Define input_image as an instance variable
        # self.input_data = None

        # self.pixmap_size = None

    def browse_image(self, input_image, scene):
        # Open a file dialog to select an image file
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.webp *.png *.jpeg *.avif)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]

            # Load the selected image file using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_img = image

            # Convert the image to QImage
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.input_data = image_rgb
            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(q_image)

            # Display the QPixmap on scene
            self.display_qpixmap(pixmap, scene)

            # Reset the view's matrix
            scene.views()[0].resetTransform()
            # Center the scene in the view
            scene.views()[0].setAlignment(Qt.AlignCenter)
            # Fit the view
            scene.views()[0].fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            # Store the size of the original pixmap
            self.pixmap_size = pixmap.size()

            # Set the input_image or target_image based on mode
            if scene == self.input_scene:
                self.input_image_rgb = image_rgb
                self.input_image=image
            elif scene == self.input_scene2:
                self.target_image_rgb = image_rgb
                self.target_image=image


    def browse_input_image(self):
        if self.tabWidget_2.currentIndex() == 2:
            self.browse_image(self.input_img_ncc, self.input_scene)
        elif self.tabWidget_2.currentIndex() == 1:
            self.browse_image(self.sift1_graphicsView, self.sift1_scene)
        elif self.tabWidget_2.currentIndex() == 0:
            self.browse_image(self.graphicsView, self.input_harris_scene)

    def browse_target_image(self):
        self.browse_image(self.input_target_ncc,self.input_scene2)
        # self.normalized_cross_correlation()
        self.feature_detection()

    def display_qpixmap(self, pixmap, scene):
        # Display the QPixmap on scene
        scene.clear()  # Clear the scene
        scene.addPixmap(pixmap)  # Add the pixmap to the scene
        scene.setSceneRect(QRectF(pixmap.rect())) # Update the scene's bounding rectangle

        # --------------------------------Display Image--------------------------------
    def display_filtered_img(self, img, graph_name, graph_scene):
        """
        Displays a filtered image in a QGraphicsView within a specified scene.

        Args:
            img (numpy.ndarray): Input image (grayscale or color).
            graph_name (QGraphicsView): QGraphicsView widget to display the image.
            graph_scene (QGraphicsScene): QGraphicsScene associated with the QGraphicsView.

        Returns:
            None
        """
        print("mama")
        if len(img.shape) == 2:
            # Grayscale image
            height, width = img.shape
            bytes_per_line = width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color image
            height, width, _ = img.shape
            bytes_per_line = 3 * width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_BGR888)

        pixmap = QPixmap.fromImage(q_image)

        pixmap = pixmap.scaled(self.pixmap_size, Qt.AspectRatioMode.KeepAspectRatio)

        graph_scene.clear()
        graph_scene.addPixmap(pixmap)
        print(f"check scene: {graph_name.scene().items()}")

        # Set the scene's dimensions to match the dimensions of the original pixmap
        graph_scene.setSceneRect(0, 0, self.pixmap_size.width(), self.pixmap_size.height())

        # Resize the QGraphicsView to fit the scene
        graph_name.fitInView(graph_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)



    def browse(self):
        """
        Opens a file dialog to select an image file and displays it.
        """
        self.flag = 1
        # Open a file dialog to select an image file
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]

            # Load the selected image using OpenCV
            image = cv2.imread(file_path)

            if image is None:
                print("Error: Unable to load image")
                return

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to QImage
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.input_data = image
            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(q_image)

            # Clear the scene and add the pixmap to the scene
            self.input_scene_harris.clear()
            self.input_scene_harris.addPixmap(pixmap)

            # Set scene rect to match image dimensions
            self.input_scene_harris.setSceneRect(0, 0, width, height)




            # Center the view
            self.graphicsView.setAlignment(Qt.AlignCenter)

    def apply_harris(self):
        """
        This function applies the Harris operator to the image data. The threshold for the operator is set by `self.harris_threshold`.
        """
        self.apply_harris_operator(threshold=self.harris_threshold)

    def find_local_maxima(self, binary_img, neighborhood):
        """
        This function finds local maxima in the binary image.

        Parameters:
            binary_img: Binary image containing corners.
            neighborhood: The neighborhood mask for local maximum search.

        Returns:
            local_maxima: Array containing ones in places of local maxima and zeros elsewhere.
        """
        local_maxima = np.zeros_like(binary_img)

        # Iterate over the image, except for borders
        for i in range(1, binary_img.shape[0] - 1):
            for j in range(1, binary_img.shape[1] - 1):
                if binary_img[i, j] == 1:
                    # Check if the current pixel is greater than or equal to its neighbors
                    if np.all(binary_img[i - 1:i + 2, j - 1:j + 2] <= binary_img[i, j]):
                        local_maxima[i, j] = 1
                j=j+2

            i = i +2

        return local_maxima

    def apply_harris_operator(self, threshold):
        """
        This function applies the Harris corner detection operator to the image data.

        Parameters:
            threshold: The threshold for the Harris operator.
        """
        t_start = time.time()

        # Keep a copy of the original image
        img_copy = self.input_data.copy()

        # Convert the BGR image to grayscale
        gray_img = cv2.cvtColor(self.input_data, cv2.COLOR_BGR2GRAY)

        # Compute derivatives
        deriv_x, deriv_y = self.compute_gauss_derivatives(3)  # Example call with size=3

        # Convolve image with derivative kernels
        conv_x = signal.convolve(gray_img, deriv_x, mode='same')
        conv_y = signal.convolve(gray_img, deriv_y, mode='same')

        # Compute structure tensor components
        gauss_kernel = self.compute_gauss_kernel(3)
        tensor_xx = signal.convolve(conv_x * conv_x, gauss_kernel, mode='same')
        tensor_xy = signal.convolve(conv_x * conv_y, gauss_kernel, mode='same')
        tensor_yy = signal.convolve(conv_y * conv_y, gauss_kernel, mode='same')

        # Harris response
        determinant = tensor_xx * tensor_yy - tensor_xy ** 2
        trace = tensor_xx + tensor_yy
        f_value = (determinant / (trace+1e-10))

        # Thresholding
        threshold = np.max(f_value) * threshold
        binary_img = (f_value > threshold) * 1

        # # Non-maximum suppression
        # Non-maximum suppression
        # Non-maximum suppression
        neighborhood = np.ones((3, 3))

        local_max = self.find_local_maxima(binary_img, neighborhood)
        binary_img = local_max

        # # Get corner coordinates
        detected_corners = np.argwhere(binary_img > 0)
        print(f"number of detected corners = {len(detected_corners)}")

        # Draw corners on the original image
        for y, x in detected_corners:
            cv2.circle(img_copy, (x, y), 3, (255, 0, 0), -1)  # Draw a blue circle at each corner
        print("cv2.circle")
        # Convert the modified original image to QImage
        height, width, _ = img_copy.shape
        bytes_per_line = 3 * width  # For color image
        q_image = QImage(img_copy.data, width, height, bytes_per_line, QImage.Format_RGB888)
        print("q_image")
        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(q_image)

        # Display the QPixmap on self.output_view
        self.output_harris_scene.clear()  # Clear the scene
        # Display the QPixmap on scene
        self.display_qpixmap(pixmap, self.output_harris_scene)
        self.output_harris_scene.views()[0].resetTransform()
        # Center the scene in the view
        self.output_harris_scene.views()[0].setAlignment(Qt.AlignCenter)
        # Fit the view
        self.output_harris_scene.views()[0].fitInView(self.output_harris_scene.sceneRect(), Qt.KeepAspectRatio)



        print("AlignCenter")
        t_end = time.time()
        harris_time = t_end - t_start
        print(f"time for applying the harris operator : {harris_time}")


    def compute_gauss_derivatives(self, sizex, sizey=None):
        """
        This function computes the x and y derivatives of a 2D Gaussian kernel.

        Parameters:
            sizex: The size of the kernel in the x direction.
            sizey: The size of the kernel in the y direction. If not provided, it defaults to `sizex`.

        Returns:
            gx, gy: The x and y derivatives of the 2D Gaussian kernel.
        """
        sizex = int(sizex)
        sizey = int(sizey) if sizey else sizex

        y, x = np.mgrid[-sizex:sizex + 1, -sizey:sizey + 1]

        factor_x = float((0.5 * sizex) ** 2)
        factor_y = float((0.5 * sizey) ** 2)

        exp_component = np.exp(-(x ** 2 / factor_x + y ** 2 / factor_y))

        gx = -x * exp_component
        gy = -y * exp_component

        return gx, gy

    def compute_gauss_kernel(self, sizex, sizey=None):
        """
        This function computes a 2D Gaussian kernel.

        Parameters:
            sizex: The size of the kernel in the x direction.
            sizey: The size of the kernel in the y direction. If not provided, it defaults to `sizex`.

        Returns:
            g: The computed 2D Gaussian kernel.
        """
        sizex = int(sizex)
        sizey = int(sizey) if sizey else sizex

        x, y = np.mgrid[-sizex:sizex + 1, -sizey:sizey + 1]

        g = np.exp(-(x ** 2 / float(sizex) + y ** 2 / float(sizey)))

        return g / g.sum()     

    def normalized_cross_correlation(self,image, template):

        """
           Computes normalized cross-correlation (NCC) between an image and a template.

           Args:
               image (numpy.ndarray): Input image.
               template (numpy.ndarray): Template image.

           Returns:
               numpy.ndarray: Image with matches drawn.
        """
        # Step 1: Feature extraction (SIFT)
        num_matches = 0  # Initialize a counter for the number of matches
         # Create a SIFT object

        # Extract keypoints and descriptors for each image
        keypoints1, descriptors1 = self.sift.detectAndCompute(image,
                                                         None)  # Extract keypoints and descriptors for the input image
        keypoints2, descriptors2 = self.sift.detectAndCompute(template,
                                                         None)  # Extract keypoints and descriptors for the template image

        # Step 2: Feature Matching
        # Initialize lists to store matching results
        matches_ncc = []  # List to store NCC matches

        # Loop through all descriptors in image 1
        for idx1, descriptor1 in enumerate(descriptors1):
            best_match_ncc = None  # Initialize the best NCC match
            best_ncc_value = -float('inf')  # Initialize the best NCC value

            # Loop through all descriptors in image 2
            for idx2, descriptor2 in enumerate(descriptors2):
                # Compute NCC
                current_ncc = np.sum((descriptor1 - np.mean(descriptor1)) * (descriptor2 - np.mean(descriptor2))) / (
                        np.std(descriptor1) * np.std(descriptor2))

                # Update NCC best match
                if current_ncc > best_ncc_value:
                    best_ncc_value = current_ncc  # Update the best NCC value
                    best_match_ncc = cv2.DMatch(idx1, idx2, 0, 1 - current_ncc)  # Create a new NCC match object

            if best_match_ncc is not None:
                num_matches += 1  # Increment the number of matches
            # Append best NCC match to the list
            matches_ncc.append(best_match_ncc)  # Add the best NCC match to the list

            matches_ncc.sort(key=lambda x: x.distance)
            # Select top 20 matches
            matches_ncc = matches_ncc[:20]  # Keep only the top 20 matches

        print("Number of matches for NCC:", num_matches)  # Print the number of matches

        # NCC matching visualization
        image_ncc = cv2.drawMatches(image, keypoints1, template, keypoints2, matches_ncc, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # Draw the matches between the input image and the template

        return image_ncc  # Return the image with matches drawn



        # # Compute mean and standard deviation of the template
        #
        # template_mean = np.mean(template)
        # template_std = np.std(template)
        #
        # # Compute the size of the template
        # template_height, template_width = template.shape[:2]
        #
        # # Initialize the correlation map
        # correlation_map = np.zeros((image.shape[0] - template_height + 1,
        #                             image.shape[1] - template_width + 1))
        #
        # # Iterate over all possible positions in the image
        # for y in range(correlation_map.shape[0]):
        #     for x in range(correlation_map.shape[1]):
        #         # Get the image patch
        #         patch = image[y:y + template_height, x:x + template_width]
        #
        #         # Compute mean and standard deviation of the image patch
        #         patch_mean = np.mean(patch)
        #         patch_std = np.std(patch)
        #
        #         # Compute the cross-correlation coefficient
        #         correlation = np.sum((patch - patch_mean) * (template - template_mean)) / (patch_std * template_std)
        #
        #         # Normalize the correlation coefficient
        #         correlation_map[y, x] = correlation / (template_height * template_width)
        #
        # return correlation_map

    def feature_detection(self):
        """
            Perform feature detection using either Sum of Square Differences (SSD) or Normalized Cross-Correlation (NCC).
        """
        start_time = time.time()  # Record the start time

        mode = self.feature_comboBox.currentText()  # Get the selected mode from the combo box
        if mode == "SSD":  # If SSD mode is selected
            matched_img = self.sum_square_diffrence(self.input_image_rgb, self.target_image_rgb)  # Perform SSD matching
        if mode == "NCC":  # If NCC mode is selected
            matched_img = self.normalized_cross_correlation(self.input_image_rgb,
                                                            self.target_image_rgb)  # Perform NCC matching

        end_time = time.time()  # Record the end time
        computation_time = end_time - start_time  # Calculate the computation time

        print(f"Matching computation time: {computation_time} seconds")  # Print the computation time
        cv2.imwrite("matched_img.jpg", matched_img)  # Save the matched image

        self.display_filtered_img(matched_img, self.out_ncc, self.output_scene)  # Display the matched image

        # correlation_map = self.normalized_cross_correlation(self.input_image_rgb, self.target_image_rgb)
        # # Find the position with the highest correlation
        # max_position = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        #
        #
        # top_left = max_position[::-1]  # Convert (row, col) to (x, y) coordinates
        # bottom_right = (top_left[0] + self.target_image_rgb.shape[1], top_left[1] + self.target_image_rgb.shape[0])
        #
        # # Blend the target image with transparency onto the original image
        # overlay = self.input_image.copy()
        # print("mariam2")
        # alpha = 0.5  # Transparency level
        # cv2.rectangle(overlay, top_left, bottom_right, (255, 0, 255), -1)  # Fill rectangle with green color
        # print("mariam3")
        # # cv2.addWeighted(overlay, alpha, self.input_image, 1 - alpha, 0, self.input_image)
        # blended_image = cv2.addWeighted(overlay, alpha, self.input_image, 1 - alpha, 0, self.input_image)
        #
        # # Save the blended image to a file
        # save_path = 'blended_image.png'
        # cv2.imwrite(save_path, blended_image)
        #
        # # Read the saved image file
        # result_image = cv2.imread(save_path)
        #
        # # Display the result in the QGraphicsView
        # self.display_filtered_img(result_image, self.out_ncc, self.output_scene)

    def sum_square_diffrence(self, image1, image2):
        """
            Computes Sum of Square Differences (SSD) between two images.

            Args:
                image1 (numpy.ndarray): First input image.
                image2 (numpy.ndarray): Second input image.

            Returns:
                numpy.ndarray: Image with matches drawn.
        """
        # Create a SIFT object
        num_matches = 0  # Initialize a counter for the number of matches

        # Extract keypoints and descriptors for each image
        keypoints1, descriptors1 = self.sift.detectAndCompute(image1,
                                                         None)  # Extract keypoints and descriptors for the first image
        keypoints2, descriptors2 = self.sift.detectAndCompute(image2,
                                                         None)  # Extract keypoints and descriptors for the second image

        matches_ssd = []  # List to store SSD matches

        # Loop through all descriptors in image 1
        for idx1, descriptor1 in enumerate(descriptors1):
            best_match_ssd = None  # Initialize the best SSD match
            best_ssd_value = float('inf')  # Initialize the best SSD value

            # Loop through all descriptors in image 2
            for idx2, descriptor2 in enumerate(descriptors2):
                # Compute SSD
                current_ssd = np.sum((descriptor1 - descriptor2) ** 2)  # Compute the SSD between descriptors
                # Update SSD best match
                if current_ssd < best_ssd_value:
                    best_ssd_value = current_ssd  # Update the best SSD value
                    best_match_ssd = cv2.DMatch(idx1, idx2, 0, current_ssd)  # Create a new SSD match object

            if best_match_ssd is not None:
                # Add best matches to list
                matches_ssd.append(best_match_ssd)

            # Sort matches based on SSD value
        matches_ssd.sort(key=lambda x: x.distance)  # Sort matches by distance

        # Select top 20 matches
        matches_ssd = matches_ssd[:20]

        # Draw top matches
        image_ssd = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_ssd, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # Draw the matches between the two images

        return image_ssd  # Return the image with matches drawn

        # correlation_map = self.normalized_cross_correlation(self.input_image_rgb, self.target_image_rgb)
        # # Find the position with the highest correlation
        # max_position = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        #
        #
        # top_left = max_position[::-1]  # Convert (row, col) to (x, y) coordinates
        # bottom_right = (top_left[0] + self.target_image_rgb.shape[1], top_left[1] + self.target_image_rgb.shape[0])
        #
        # # Blend the target image with transparency onto the original image
        # overlay = self.input_image.copy()
        # print("mariam2")
        # alpha = 0.5  # Transparency level
        # cv2.rectangle(overlay, top_left, bottom_right, (255, 0, 255), -1)  # Fill rectangle with green color
        # print("mariam3")
        # # cv2.addWeighted(overlay, alpha, self.input_image, 1 - alpha, 0, self.input_image)
        # blended_image = cv2.addWeighted(overlay, alpha, self.input_image, 1 - alpha, 0, self.input_image)
        #
        # # Save the blended image to a file
        # save_path = 'blended_image.png'
        # cv2.imwrite(save_path, blended_image)
        #
        # # Read the saved image file
        # result_image = cv2.imread(save_path)
        #
        # # Display the result in the QGraphicsView
        # self.display_filtered_img(result_image, self.out_ncc, self.output_scene)

    def lambda_minus_corner_detection(self, corner_color=[0, 0, 255], blockSize = 2, ksize = 3):
        """
        Detects Harris corners in an image and marks them with a specified color.

        Args:
            corner_color (list, optional): RGB color to mark the corners. Defaults to [0, 0, 255] (Red).
            blockSize (int, optional): Neighborhood size for computing the Harris corner response. Defaults to 2.
            ksize (int, optional): Aperture size for Sobel operators. Defaults to 3.

        Returns:
            None

        This function works by first converting the image to grayscale if it's not already. It then computes the
        derivatives Ix and Iy using the Sobel operator. The products of these derivatives are computed and summed over
        a local window. The determinant and trace of the matrix are computed, and these are used to compute the minimum
        eigenvalue (Lambda Minus) of the 2x2 structure tensor. The function then finds the indices where Lambda Minus is
        greater than a threshold. These indices represent the corners in the image.

        If the image is an RGB image, the corners are marked with the user-specified color. If the image is a grayscale
        image, the corners are marked with the mean intensity of the user-specified color.

        The function also measures the computation time for calculating Lambda Minus and prints it out.
        """
        # Start the timer to measure computation time
        start_time = time.time()

        # Create a copy of the image to avoid modifying the original image
        image = self.current_img.copy()

        # Check if the image is grayscale or color
        if len(image.shape) == 3:
            # If the image is color, convert it to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # If the image is already grayscale, use it as is
            gray = image

        # Compute the x and y derivatives using the Sobel operator
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        # Compute the products of the derivatives
        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix*Iy

        # Compute the sum of the products of the derivatives over a local window
        Sxx = cv2.boxFilter(Ixx, -1, (blockSize, blockSize))
        Syy = cv2.boxFilter(Iyy, -1, (blockSize, blockSize))
        Sxy = cv2.boxFilter(Ixy, -1, (blockSize, blockSize))

        # Compute the determinant and trace of the matrix
        det = Sxx * Syy - Sxy**2
        trace = Sxx + Syy

        # Compute the minimum eigenvalue (Lambda Minus) of the 2x2 structure tensor
        lambda_minus = trace / 2 - np.sqrt((trace / 2)**2 - det)

        # Find the indices where Lambda Minus is greater than a threshold
        indices = np.where(lambda_minus > self.harris_threshold * lambda_minus.max())
        indices = indices[0], indices[1]

        # Check if the image is grayscale or color
        if len(image.shape) == 3:
            # If the image is RGBA, convert it to RGB
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            # Mark the corners with the user-specified color
            image[indices[0], indices[1], :] = corner_color
        else:
            # If the image is grayscale, mark the corners with the mean intensity of the user-specified color
            image[indices[0], indices[1]] = np.mean(corner_color)

        # Stop the timer and compute the computation time
        end_time = time.time()
        computation_time = end_time - start_time

        self.display_filtered_img(image, self.graphicsView_2, self.output_harris_scene)
        print(f"Computation time: {computation_time}")
        
    # --------------------------------SIFT--------------------------------
    def apply_sift(self):
        start_time = time.time()
        if self.keypoints_radioButton.isChecked():
            apply_sift.sift_image("keypoints_radioButton")
            combined_keypoints_image = cv2.imread('combined_keypoints_vertical.jpg')
            self.display_filtered_img(combined_keypoints_image, self.sift_res_graphicsView, self.sift_res_scene)
        elif self.matching_radioButton.isChecked():
            apply_sift.sift_image("matching_radioButton")
            sift_image = cv2.imread('sift.jpeg')
            self.display_filtered_img(sift_image, self.sift_res_graphicsView, self.sift_res_scene)
        else:
            # Display an error message
            QMessageBox.warning(self, "Error", "Please select one of the radio buttons.")
        end_time = time.time()
        computation_time = end_time - start_time
        print("computation_time: ", computation_time)
        
        




def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()