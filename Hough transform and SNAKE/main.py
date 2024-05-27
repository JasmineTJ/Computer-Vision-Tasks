import timeit
import cv2
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
import cv2
import copy
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from libs import Contour

###################################
import numpy as np
import cv2


# import pyqtgraph

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=20, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r'GUI.ui', self)
        # ui setup
        self.handle_ui()

        self.choose_mode()

        self.radioButton.setChecked(True)

        self.mode_name_comboBox.currentIndexChanged.connect(self.choose_mode)

        self.pushButton.clicked.connect(self.browse_image)

        self.pushButton_2.clicked.connect(self.handle_apply_button)

        # Map sliders to labels
        self.slider_label_map = {
            self.sigma_slider: self.slider_sig_label,
            self.horizontalSlider_2: self.label_12,
            self.horizontalSlider_3: self.label_13,
            self.horizontalSlider_4: self.label_14,
            self.horizontalSlider_5: self.label_15,
            self.horizontalSlider_6: self.label_16,
        }

        # Connect each slider to update_label
        for slider, label in self.slider_label_map.items():
            slider.valueChanged.connect(lambda value, label=label: self.update_label(value, label))

    def update_label(self, value, label):
        # Update the label text with the slider's current value
        label.setText(str(value))

    def handle_apply_button(self):
        # Check if there is an image loaded
        if self.input_scene.items():
            self.mode = self.mode_name_comboBox.currentText()
            if self.mode == "Canny Edge Detector":
                self.edge_detection()
            if self.mode == "Snake":
                self.active_contour_model()
            if self.mode == "Line Detection":
                self.hough_lines()
            if self.mode == "Circle Detection":
                self.apply_circle_detection()
            if self.mode == "Elipse Detection":
                self.hough_ellipses()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No image loaded. Please select an image.")
            msg.setWindowTitle("Error")
            # Adjust the width of the message box to fit the text
            msg.layout().setSizeConstraint(QLayout.SetFixedSize)
            msg.exec_()

    def handle_ui(self):
        # Create a QGraphicsScene
        self.input_scene = QtWidgets.QGraphicsScene()
        self.input_image.setScene(self.input_scene)

        self.output_scene = QtWidgets.QGraphicsScene()
        self.output_image.setScene(self.output_scene)

        # Define input_image as an instance variable
        self.input_data = None

        # Define noisy_image as an instance variable
        self.noisy_image = None

        self.pixmap_size = None

        colors = ["Red", "Green", "Blue"]
        self.colors_combo.addItems(colors)

    def browse_image(self):
        # Open a file dialog to select an image file
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.webp *.png *.jpeg *.avif)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]

            # Load the selected image file using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            self.image=image
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

            # Display the QPixmap on self.graphicsView
            self.input_scene.clear()  # Clear the scene
            self.output_scene.clear()
            self.input_scene.addPixmap(pixmap)  # Add the pixmap to the scene
            self.input_scene.setSceneRect(QRectF(pixmap.rect()))  # Update the scene's bounding rectangle
            self.input_image.resetTransform()  # Reset the view's matrix
            self.input_image.setAlignment(Qt.AlignCenter)  # Center the scene in the view
            self.input_image.fitInView(self.input_scene.sceneRect(), Qt.KeepAspectRatio)  # Fit the view
            # Store the size of the original pixmap
            self.pixmap_size = pixmap.size()

    def hough_lines(self):
        """
           Detects lines in the input image using Hough transform.
           """

        color = self.colors_combo.currentText()
        print(f"color : {color}")
        peaks = self.horizontalSlider_2.value()
        print(f"peaks : {peaks}")
        T_low = self.horizontalSlider_3.value() / 100
        print(f"T_low : {T_low}")
        T_high = self.horizontalSlider_4.value() / 100
        print(f"T_high : {T_high}")
        neighborhood_size = self.horizontalSlider_5.value()
        print(f"neighborhood_size : {neighborhood_size}")

        src = np.copy(self.input_data)
        H, thetas, rhos = self.line_detection(src, T_low, T_high)
        indicies, H = self.hough_peaks(H, peaks, neighborhood_size)
        self.hough_lines_draw(color, src, indicies, rhos, thetas)

    def line_detection(self, image, T_low, T_upper):
        """
           Performs line detection on the input image using Canny edge detection and Hough transform.
           Args:
               image (numpy.ndarray): Input image.
               T_low (float): Low threshold for Canny edge detection.
               T_upper (float): High threshold for Canny edge detection.
           Returns:
               tuple: Accumulator array, array of theta values, array of rho values.
           """
        print("image")
        print(image.shape)
        grayImg = cv2.cvtColor(self.input_data, cv2.COLOR_BGR2GRAY)
        blurImg = cv2.GaussianBlur(grayImg, (5, 5), 1.5)
        edgeImg = cv2.Canny(blurImg, T_low, T_upper)
        # edgeImg = cv2.imread('canny.jpg', cv2.IMREAD_GRAYSCALE)

        height, width = edgeImg.shape

        maxDist = int(np.around(np.sqrt(height ** 2 + width ** 2)))

        thetas = np.deg2rad(np.arange(-90, 90))
        rhos = np.linspace(-maxDist, maxDist, 2 * maxDist)

        accumulator = np.zeros((2 * maxDist, len(thetas)))

        for y in range(height):
            for x in range(width):
                if edgeImg[y, x] > 0:
                    for k in range(len(thetas)):
                        r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                        accumulator[int(r) + maxDist, k] += 1

        return accumulator, thetas, rhos

    def hough_peaks(self, H, peaks, neighborhood_size=3):
        """
            Finds peaks in the Hough accumulator array.
            Args:
                H (numpy.ndarray): Hough accumulator array.
                peaks (int): Number of peaks to find.
                neighborhood_size (int, optional): Size of the neighborhood to detect peaks. Defaults to 3.
            Returns:
                tuple: List of peak indices, modified Hough accumulator array.
            """

        indices = []
        H1 = np.copy(H)

        # loop through number of peaks to identify
        for i in range(peaks):
            idx = np.argmax(H1)  # find argmax in flattened array
            H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H to be 2d array
            indices.append(H1_idx)

            idx_y, idx_x = H1_idx  # separate x, y indices from argmax(H)

            # if idx_x is too close to the edges choose appropriate values
            if (idx_x - (neighborhood_size / 2)) < 0:
                min_x = 0
            else:
                min_x = idx_x - (neighborhood_size / 2)
            if (idx_x + (neighborhood_size / 2) + 1) > H.shape[1]:
                max_x = H.shape[1]
            else:
                max_x = idx_x + (neighborhood_size / 2) + 1

            # if idx_y is too close to the edges choose appropriate values
            if (idx_y - (neighborhood_size / 2)) < 0:
                min_y = 0
            else:
                min_y = idx_y - (neighborhood_size / 2)
            if (idx_y + (neighborhood_size / 2) + 1) > H.shape[0]:
                max_y = H.shape[0]
            else:
                max_y = idx_y + (neighborhood_size / 2) + 1

            # bound each index by the neighborhood size and set all values to 0
            for x in range(int(min_x), int(max_x)):
                for y in range(int(min_y), int(max_y)):

                    # remove neighborhoods in H1
                    H1[y, x] = 0

                    # highlight peaks in original H
                    if x == min_x or x == (max_x - 1):
                        H[y, x] = 255
                    if y == min_y or y == (max_y - 1):
                        H[y, x] = 255

        # return the indices and the original Hough space with selected points
        return indices, H

    def hough_lines_draw(self, color, img, indices, rhos, thetas):
        print("hough draw entered")

        result_img = np.copy(img)  # Create a copy of the input image to draw lines on
        print("number of detected lines")
        print(len(indices))

        for i in range(len(indices)):
            # get lines from rhos and thetas
            rho = rhos[indices[i][0]]
            theta = thetas[indices[i][1]]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # these are then scaled so that the lines go off the edges of the image
            y1 = int(y0 + 1000 * (a))
            x1 = int(x0 + 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            if color == 'Red':
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif color == 'Blue':
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 0, 205), 2)
            elif color == 'Green':
                cv2.line(result_img, (x1, y1), (x2, y2), (50, 205, 50), 2)

        # Convert the modified image to QPixmap
        height, width, channel = result_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Clear the output scene and add the QPixmap
        self.output_scene.clear()
        self.output_scene.addPixmap(pixmap)
        self.output_image.fitInView(self.output_scene.sceneRect(), Qt.KeepAspectRatio)

    #####################################################################################################################
    def choose_mode(self):
        self.mode = self.mode_name_comboBox.currentText()
        labels = [getattr(self, f'label_{i + 6}') for i in range(11)]
        sliders = [getattr(self, f'horizontalSlider_{i + 2}') for i in range(5)]
        edgelist = [self.sigma_slider, self.kernal_combo, self.sigma_label, self.slider_sig_label, self.kernal_label]

        if self.mode == "Canny Edge Detector":
            for label in labels:
                label.hide()
            for slider in sliders:
                slider.setMinimum(0)  # Set the minimum value of the slider to 0
                slider.setMaximum(100)  # Set the maximum value of the slider to 100
                slider.hide()
            self.colors_combo.hide()
            for i in edgelist:
                i.show()
            self.radioButton.hide()
            self.radioButton_2.hide()
            self.label_4.show()
            self.label_3.show()
            self.lineEdit_low.show()
            self.lineEdit_high.show()

        if self.mode == "Line Detection":
            textlst = ["Number of lines", "Low Threshold", "High Threshold", "Neighbor Size", "Lines Color"]
            for label, text in zip(labels, textlst):
                label.setText(text)
            for i in edgelist:
                i.hide()
            self.horizontalSlider_6.hide()
            self.label_11.hide()
            self.label_16.hide()
            self.label_4.hide()
            self.label_3.hide()
            self.lineEdit_low.hide()
            self.lineEdit_high.hide()
            self.show_lines_and_circles()
            for slider in sliders:
                slider.setMinimum(0)  # Set the minimum value of the slider to 0
                slider.setMaximum(100)  # Set the maximum value of the slider to 100
                slider.setValue(0)  # Set the current value of the slider to 0
            self.radioButton.hide()
            self.radioButton_2.hide()

        if self.mode == "Circle Detection":
            textlst = ["Min Radius", "Max Radius", "Bin Threshold", "Pixel Threshold", "Lines Color"]
            for label, text in zip(labels, textlst):
                label.setText(text)
            for i in edgelist:
                i.hide()
            self.horizontalSlider_6.hide()
            self.label_11.hide()
            self.label_16.hide()
            self.label_4.hide()
            self.label_3.hide()
            self.lineEdit_low.hide()
            self.lineEdit_high.hide()
            self.show_lines_and_circles()
            self.radioButton.hide()
            self.radioButton_2.hide()
            # set_sliders:
            self.set_sliders(self.horizontalSlider_2)
            # self.set_sliders(self.horizontalSlider_3)
            # self.set_sliders(self.horizontalSlider_4)
            self.set_sliders(self.horizontalSlider_5)
            self.horizontalSlider_4.setMinimum(0)
            self.horizontalSlider_4.setMaximum(10)  # Maximum value represents 1 (since steps are 0.1)
            self.horizontalSlider_4.setSingleStep(1)  # Each step represents 0.1
            self.horizontalSlider_4.valueChanged.connect(lambda value: self.label_14.setText(f"{value / 10.0}"))
            self.horizontalSlider_3.setMinimum(50)
            self.horizontalSlider_3.setMaximum(250)  # Maximum value represents 1 (since steps are 0.1)
            # self.horizontalSlider_3.setSingleStep(1)  # Each step represents 0.1
        if self.mode == "Elipse Detection":
            for label in labels:
                label.hide()
            for slider in sliders:
                slider.hide()
            for i in edgelist:
                i.hide()
            self.horizontalSlider_2.show()
            self.horizontalSlider_3.show()
            self.horizontalSlider_4.show()
            self.label_14.show()
            self.label_12.show()
            self.label_13.show()
            self.label_6.show()
            self.label_7.show()
            self.label_8.show()
            self.label_6.setText("max size")
            self.label_7.setText("min size")
            self.label_8.setText("voting threshold")
            self.label_10.setText("Lines color")
            self.horizontalSlider_2.setMinimum(2)
            self.horizontalSlider_2.setMaximum(10)
            self.horizontalSlider_4.setMinimum(20)
            self.horizontalSlider_4.setMaximum(300)  # Maximum value represents 1 (since steps are 0.1)
            self.horizontalSlider_3.setMinimum(1)
            self.horizontalSlider_3.setMaximum(5)


        if self.mode == "Snake":
            textlst = ["Alpha", "Beta", "Gamma", "Iterations"]
            for label, text in zip(labels, textlst):
                label.setText(text)
            for i in edgelist:
                i.hide()
            self.label_11.hide()
            self.lineEdit_low.hide()
            self.lineEdit_high.hide()
            self.label_4.hide()
            self.label_3.hide()
            self.radioButton.show()
            self.radioButton_2.show()
            self.show_snake()
            for slider in sliders:
                slider.setMinimum(0)  # Set the minimum value of the slider to 0
                slider.setMaximum(100)  # Set the maximum value of the slider to 100
                slider.setValue(0)  # Set the current value of the slider to 0

    def show_lines_and_circles(self):
        labels = [getattr(self, f'label_{i + 6}') for i in range(11)]
        sliders = [getattr(self, f'horizontalSlider_{i + 2}') for i in range(5)]
        edgelist = [self.sigma_slider, self.kernal_combo, self.sigma_label, self.slider_sig_label, self.kernal_label]
        for label in labels:
            label.show()
        for slider in sliders:
            slider.show()
        self.colors_combo.show()
        self.horizontalSlider_6.hide()
        self.label_11.hide()
        self.label_16.hide()

    def show_snake(self):
        self.show_lines_and_circles()
        self.horizontalSlider_6.hide()
        self.label_11.hide()
        self.label_16.hide()
        self.label_10.hide()
        self.colors_combo.hide()
        
    # --------------------------------Ellipse Hough Transform --------------------------------
        
    def hough_ellipses(self):
        """
           Detects ellipse in the input image using Hough transform.
           """
        
    
        max_size = self.horizontalSlider_2.value()
        min_size = self.horizontalSlider_3.value() 
        threshold = self.horizontalSlider_4.value() 

        src = np.copy(self.input_data)
        ellipses = self.detect_ellipses(src, max_size, min_size, threshold)
        self.draw_ellipes(src, ellipses)

    def detect_ellipses(self,image, max_size, min_size, threshold):
        """
        Detect ellipses in an image using the Hough Transform.

        Args:
            image (numpy.ndarray): Input image.
            max_size (int): Maximum size of the detected ellipse.
            min_size (int): Minimum size of the detected ellipse.
            threshold (int): Threshold for voting.

        Returns:
            list: List of tuples containing parameters for detected ellipses (center x, center y, major axis, minor axis).
        """
        gray = cv2.cvtColor(self.input_data, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Initialize accumulator
        accumulator = np.zeros((gray.shape[0], gray.shape[1], max_size - min_size))

        # Iterate over edge pixels
        for y in range(edges.shape[0]):
            for x in range(edges.shape[1]):
                if edges[y, x] > 0:
                    # Vote for ellipses passing through this point
                    for a in range(min_size, max_size):
                        for b in range(min_size, max_size):
                            for theta in range(0, 360):
                                rad = np.deg2rad(theta)
                                x0 = int(x - a * np.cos(rad))
                                y0 = int(y - b * np.sin(rad))
                                if x0 >= 0 and x0 < gray.shape[1] and y0 >= 0 and y0 < gray.shape[0]:
                                    accumulator[y0, x0, a - min_size] += 1

        # Find ellipses with enough votes
        ellipses = []
        for a in range(min_size, max_size):
            for y in range(edges.shape[0]):
                for x in range(edges.shape[1]):
                    if accumulator[y, x, a - min_size] > threshold:
                        ellipses.append((x, y, a, a))

        return ellipses

    def draw_ellipes(self, img, ellipses):
        """
        Draw ellipses on an image.

        Args:
            color (str): Color of the ellipses ('Red', 'Blue', or 'Green').
            img (numpy.ndarray): Input image.
            ellipses (list): List of tuples containing parameters for ellipses (center x, center y, major axis, minor axis).

        Returns:
            None
        """
        print("hough draw entered")

        result_img = np.copy(img)  # Create a copy of the input image to draw lines on
        for ellipse in ellipses:
            cv2.ellipse(result_img, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), 0, 0, 360, (0, 0, 255), 2)
            
        # Convert the modified image to QPixmap
        height, width, channel = result_img.shape
        bytes_per_line = 3 * width
        q_image = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Clear the output scene and add the QPixmap
        self.output_scene.clear()
        self.output_scene.addPixmap(pixmap)
        self.output_image.fitInView(self.output_scene.sceneRect(), Qt.KeepAspectRatio)
        

    # --------------------------------Rgb to Gray--------------------------------

    def rgbtogray(self, image):
        """
        Converts an RGB image to grayscale.

        Args:
            image (numpy.ndarray): Input RGB image.

        Returns:
            numpy.ndarray: Grayscale image.
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # --------------------------------Canny Filter--------------------------------
    # step 1: image in gray level
    # step 2: apply gaussian filter for noise reduction(gaussian smoothing)
    def get_gaussian_kernel(self, kernal_size, sigma):
        """
        Generates a Gaussian kernel for noise reduction (Gaussian smoothing).

        Args:
            kernal_size (int): Size of the kernel (odd number).
            sigma (float, optional): Standard deviation for Gaussian distribution. Defaults to 1.

        Returns:
            numpy.ndarray: Gaussian kernel normalized to sum to 1.
        """
        gaussian_kernal = np.zeros((kernal_size, kernal_size), np.float32)
        size = kernal_size // 2

        for x in range(-size, size + 1):
            for y in range(-size, size + 1):
                a = 1 / (2 * np.pi * (sigma ** 2))
                b = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
                gaussian_kernal[x + size, y + size] = a * b
        return gaussian_kernal / gaussian_kernal.sum()

    def apply_filtering(self, input_image, kernal):
        """
        Applies a filter to an input image using convolution.

        Args:
            input_image (numpy.ndarray): Input image.
            kernal (numpy.ndarray): Filter kernel.

        Returns:
            numpy.ndarray: Filtered output image.
        """
        output_image = []
        kernal_size = len(kernal)
        kernal_half = kernal_size // 2
        rows_count = len(input_image)
        columns_count = len(input_image[0])

        image_copy = copy.deepcopy(input_image)

        # wrap the image in the edge pixels
        for i in range(rows_count):
            for j in range(kernal_half):
                image_copy[i].insert(0, input_image[i][-1 - j])
                image_copy[i].append(input_image[i][j])
        for i in range(kernal_half):
            image_copy.append(image_copy[2 * i])
            image_copy.insert(0, image_copy[-2 - 2 * i].copy())

        # apply filtering
        new_rows_count = len(image_copy)
        new_columns_count = len(image_copy[0])

        for i in range(kernal_half, new_rows_count - kernal_half):
            output_row = []
            for j in range(kernal_half, new_columns_count - kernal_half):
                sum = 0
                for x in range(len(kernal)):
                    for y in range(len(kernal)):
                        x1 = i + x - kernal_half
                        y1 = j + y - kernal_half
                        sum += image_copy[x1][y1] * kernal[x][y]
                output_row.append(sum)
            output_image.append(output_row)

        return output_image

    # step 3 : gradient estimation(take 1st derivative)
    def gradient_estimate(self, image, gradient_estimation_filter_type):
        """
        Estimates the gradient of an input image using the specified filter type.

        Args:
            image (numpy.ndarray): Input grayscale image.
            gradient_estimation_filter_type (str): Filter type ("sobel", "prewitt", or "robert").

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Gradient magnitude image.
                - numpy.ndarray: Gradient direction (theta) image.
        """
        if (gradient_estimation_filter_type == "sobel"):
            Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
            My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        elif (gradient_estimation_filter_type == "prewitt"):
            Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
            My = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
        else:
            Mx = np.array([[0, 1], [-1, 0]], np.float32)
            My = np.array([[1, 0], [0, -1]], np.float32)

        X = self.apply_filtering(image, Mx)
        Y = self.apply_filtering(image, My)

        G = np.hypot(X, Y)
        G = G / G.max() * 255
        theta = np.arctan2(Y, X)

        return (G, theta)

    # step 4 : non-maximal suppression (thin out edges and keep only the strongest ones)
    def non_maximal_suppression(self, image, gradient_direction):
        """
        Performs non-maximal suppression on an input image based on gradient direction.

        Args:
            image (numpy.ndarray): Input gradient magnitude image.
            gradient_direction (numpy.ndarray): Gradient direction (theta) image.

        Returns:
            numpy.ndarray: Image after non-maximal suppression.
        """
        rows_count = len(image)
        columns_count = len(image[0])

        output_image = np.zeros((rows_count, columns_count), dtype=np.int32)
        theta = gradient_direction * 180. / np.pi
        theta[theta < 0] += 180

        for i in range(1, rows_count - 1):
            for j in range(1, columns_count - 1):
                next = 255
                previous = 255
                if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                    next = image[i, j + 1]
                    previous = image[i, j - 1]
                elif (22.5 <= theta[i, j] < 67.5):
                    next = image[i + 1, j - 1]
                    previous = image[i - 1, j + 1]
                elif (67.5 <= theta[i, j] < 112.5):
                    next = image[i + 1, j]
                    previous = image[i - 1, j]
                elif (112.5 <= theta[i, j] < 157.5):
                    next = image[i - 1, j - 1]
                    previous = image[i + 1, j + 1]

                if (image[i, j] >= next) and (image[i, j] >= previous):
                    output_image[i, j] = image[i, j]
                else:
                    output_image[i, j] = 0

        return output_image

    # step 5: Double Thresholding (classify pixels as strong, weak, or non-edges based on intensity thresholds)
    def double_threshold(self, image, low_threshold_ratio, high_threshold_ratio):
        """
        Applies double thresholding to an input image based on low and high threshold ratios.

        Args:
            image (numpy.ndarray): Input gradient magnitude image.
            low_threshold_ratio (float): Ratio of the low threshold relative to the maximum image value.
            high_threshold_ratio (float): Ratio of the high threshold relative to the maximum image value.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Image after double thresholding.
                - int: Weak threshold value.
                - int: Strong threshold value.
        """
        high_threshold = image.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio

        rows_count = len(image)
        columns_count = len(image[0])
        output_image = np.zeros((rows_count, columns_count), dtype=np.int32)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i = []
        strong_j = []
        weak_i = []
        weak_j = []
        for i in range(len(image)):
            for j in range(len(image[0])):
                if (image[i, j] >= high_threshold):
                    strong_i.append(i)
                    strong_j.append(j)
                if ((image[i, j] <= high_threshold) & (image[i, j] >= low_threshold)):
                    weak_i.append(i)
                    weak_j.append(j)
        strong_i = np.array(strong_i)
        strong_j = np.array(strong_j)
        weak_i = np.array(weak_i)
        weak_j = np.array(weak_j)

        output_image[strong_i, strong_j] = strong
        output_image[weak_i, weak_j] = weak

        return (output_image, weak, strong)

    # step 6: hysteresis thresholding (link weak edges to strong edges)
    def hysteresis_edge_track(self, image, weak, strong=255):
        """
        Performs edge tracking with hysteresis on an input image.

        Args:
            image (numpy.ndarray): Input image after double thresholding.
            weak (int): Weak threshold value.
            strong (int, optional): Strong threshold value. Defaults to 255.

        Returns:
            numpy.ndarray: Image after hysteresis edge tracking.
        """
        rows_count = len(image)
        columns_count = len(image[0])
        for i in range(1, rows_count - 1):
            for j in range(1, columns_count - 1):
                if (image[i, j] == weak):
                    if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (
                            image[i + 1, j + 1] == strong)
                            or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                            or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                    image[i - 1, j + 1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
        return image

    def canny(self, img, kernal_size, low_threshold_ratio, high_threshold_ratio, sigma):
        """
        Applies the Canny edge detection algorithm to an input image.

        Args:
            img (numpy.ndarray): Input grayscale image.

        Returns:
            numpy.ndarray: Image with Canny edges highlighted.
        """
        # kernal_size = 3
        # low_threshold_ratio = 0.05
        # high_threshold_ratio = 0.09
        gradient_estimation_filter_type = "sobel"

        # step 2 : apply gaussian kernal to filter noise
        kernal = self.get_gaussian_kernel(kernal_size, sigma)
        image_without_noise = self.apply_filtering(img.tolist(), kernal)

        # step 3 : gradient estimation
        assert (gradient_estimation_filter_type in ["sobel", "prewitt",
                                                    "robert"]), "gradient estimation filter type should be [\"prewitt\", \"sobel\", \"robert\"]"
        G, theta = self.gradient_estimate(image_without_noise, gradient_estimation_filter_type)

        # step 4 : non maximal suppression
        image_with_thin_edges = self.non_maximal_suppression(G, theta)

        # step 5 : double threshold and hysteresis thresholding
        final_image, weak, strong = self.double_threshold(image_with_thin_edges, low_threshold_ratio,
                                                          high_threshold_ratio)

        # edge tracking with hysteresis
        img = self.hysteresis_edge_track(final_image, weak, strong=255)
        return img

    # --------------------------------Edge Detection--------------------------------
    def edge_detection(self):
        """
        Performs canny edge detection on the currently displayed image.

        Returns:
            None
        """
        # get current original image
        curr_image = self.get_current_orignal_img()
        if self.input_scene.items():
            # get current text
            self.current_text = self.mode_name_comboBox.currentText()
            print("self.current_text:", self.current_text)
            rgbtogray_img = self.rgbtogray(curr_image)
            if self.current_text == "None":
                self.input_scene.clear()
                return None
            elif self.current_text == "Canny Edge Detector":
                # Get the chosen kernal size from the combobox
                curr_kernal = str(self.kernal_combo.currentText())
                if curr_kernal == "3 X 3 ":
                    kernal_size = 3
                elif curr_kernal == "5 X 5":
                    kernal_size = 5
                elif curr_kernal == "7 X 7":
                    kernal_size = 7
                else:
                    return None

                print("kernal size", kernal_size)

                # Read T low and T high from the line edit
                T_low = float(self.lineEdit_low.text())
                T_high = float(self.lineEdit_high.text())

                # Read the slider value of sigma
                sigma = self.sigma_slider.value()

                # Send the parameters to the canny function
                filtered_image = self.canny(rgbtogray_img, kernal_size, T_low, T_high, sigma)
                cv2.imwrite('canny.jpg', np.real(filtered_image))
                filtered_image = cv2.imread('canny.jpg', cv2.IMREAD_GRAYSCALE)
            else:
                return None

            self.display_filtered_img(filtered_image, self.output_image, self.output_scene)

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

    # --------------------------------Current Original Image--------------------------------
    def get_current_orignal_img(self):
        """
        Retrieves the currently displayed image from the QGraphicsView.

        Returns:
            numpy.ndarray or None: The image data (RGB format) if an image is loaded, or None if no image is loaded.
        """
        # Check if there is an image loaded
        if self.input_scene.items():
            # Get the currently displayed image from self.original_image
            pixmap_item = self.input_scene.items()[0]
            pixmap = pixmap_item.pixmap()
            img = pixmap.toImage()
            # Convert the image to RGB888 format (if not already)
            if img.format() != QImage.Format_RGB888:
                img = img.convertToFormat(QImage.Format_RGB888)
            width, height = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.array(ptr).reshape(height, width, 3)
            return arr
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No image loaded. Please select an image.")
            msg.setWindowTitle("Error")
            # Adjust the width of the message box to fit the text
            msg.layout().setSizeConstraint(QLayout.SetFixedSize)
            msg.exec_()
            return None

    def active_contour_model(self):
        """
        This function applies the Active Contour Model (Snake) to an image.

        The function reads an image, creates an initial square contour, calculates the external energy of the image, and then applies the active contour model for a number of iterations. The initial and final contours are plotted on the image.

        Parameters:
        None

        Returns:
        None

        Note:
        The function uses several parameters such as alpha, beta, gamma, w_line, w_edge, num_x_points, num_y_points, and num_iterations. These parameters are set using sliders in the GUI.
        """

        # Get contour parameters from the GUI sliders
        alpha = self.horizontalSlider_2.value()
        beta = self.horizontalSlider_3.value() / 100.0
        gamma = self.horizontalSlider_4.value() / 50
        num_iterations = self.horizontalSlider_5.value()
        # Set the number of points for the ellipse contour
        num_points_circle = 65
        # Set the number of points for the square contour
        num_x_points = 180
        num_y_points = 180

        # Set the weights for the line and edge energies
        w_line = 1
        w_edge = 8

        # Initial variables
        contour_x, contour_y, window_coordinates = None, None, None

        # Start the timer to calculate the function run time
        start_time = timeit.default_timer()

        # Copy the image because cv2 will edit the original source in the contour
        image_src = np.copy(self.current_img)

        if self.radioButton.isChecked():
            # Create the initial square contour and display it on the GUI
            contour_x, contour_y, window_coordinates = Contour.create_square_contour(image=image_src,
                                                                                     num_x_points=num_x_points,
                                                                                     num_y_points=num_y_points)
        elif self.radioButton_2.isChecked():
            contour_x, contour_y, window_coordinates = Contour.create_ellipse_contour(image=image_src,
                                                                                      num_points=num_points_circle)

        # Display the input image after creating the contour
        src_copy = np.copy(image_src)
        initial_image = self.draw_contour_on_image(src_copy, contour_x, contour_y)
        self.display_filtered_img(initial_image, self.output_image, self.output_scene)

        # Calculate the external energy which will be used in each iteration of the greedy algorithm
        external_energy = gamma * Contour.calculate_external_energy(image_src, w_line, w_edge)

        # Copy the coordinates to update them in the main loop
        contour_x, contour_y = np.copy(contour_x), np.copy(contour_y)

        # Apply the active contour algorithm for the specified number of iterations
        for _ in range(num_iterations):
            # Update the contour coordinates
            contour_x, contour_y = Contour.iterate_contour(image=image_src, contour_x=contour_x, contour_y=contour_y,
                                                           external_energy=external_energy,
                                                           window_coordinates=window_coordinates,
                                                           alpha=alpha, beta=beta)

            # Display the updated contour after each iteration
            src_copy = np.copy(image_src)
            processed_image = self.draw_contour_on_image(src_copy, contour_x, contour_y)
            # processed_image_BGR = self.convert_to_bgr(processed_image)
            self.display_filtered_img(processed_image, self.output_image, self.output_scene)

            # Allow the GUI to update QGraphicsView without lagging
            QtWidgets.QApplication.processEvents()

        # Convert the final contour coordinates to the format expected by calculate_chain_code
        final_contour = [(x, y) for x, y in zip(contour_x, contour_y)]

        # Calculate chain code for the final contour
        chain_code_8 = self.calculate_chain_code_8_connected(final_contour)
        print("chain code 8: ", chain_code_8)

        chain_code_4 = self.calculate_chain_code_4_connected(final_contour)
        print("chain code 4: ", chain_code_4)

        # calculate contour perimeter
        contour_perimeter = self.contour_perimeter(contour_x, contour_y)
        print(f"Contour perimeter: {contour_perimeter:.2f}")

        # calculate contour area
        contour_area = self.contour_area(len(contour_x), contour_x, contour_y)
        print(f"Contour area: {contour_area:.2f} square units")

        # Stop the timer and print the elapsed time
        end_time = timeit.default_timer()
        elapsed_time = format(end_time - start_time, '.5f')
        print(f"Elapsed time: {elapsed_time} seconds")

    @staticmethod
    def draw_contour_on_image(image: np.ndarray, points_x: np.ndarray, points_y: np.ndarray) -> np.ndarray:
        """
        Draws a contour on an image.

        Parameters:
        image (np.ndarray): The source image.
        points_x (np.ndarray): The x-coordinates of the contour points.
        points_y (np.ndarray): The y-coordinates of the contour points.

        Returns:
        np.ndarray: The image with the contour drawn on it.
        """

        # Copy the image to prevent modifying the original image
        image_copy = np.copy(image)

        # Combine the x and y coordinates into a list of points
        points = np.array(list(zip(points_x, points_y)), dtype=np.int32)

        # Reshape the points array to the required format for cv2.polylines
        points = points.reshape((-1, 1, 2))

        # Check if the image is grayscale or color
        if len(image_copy.shape) == 2 or image_copy.shape[2] == 1:
            # If the image is grayscale, draw the contour in white
            contour_color = (255, 255, 255)
        else:
            # If the image is color, draw the contour in green
            contour_color = (0, 255, 0)

        # Draw the contour on the image
        image_with_contour = cv2.polylines(image_copy, [points], isClosed=True, color=contour_color, thickness=2)

        return image_with_contour


    # --------------------------------Chain Code--------------------------------

    def calculate_chain_code_8_connected(self, contour):
        """
        Calculates the 8-connected chain code for a given contour.

        Parameters:
        contour (list): List of (x, y) tuples representing the contour points.

        Returns:
        list: List of chain code numbers.

        Note:
        Assumes that the contour is closed (i.e., starts and ends at the same point).
        """
        chain_code = []
        num_points = len(contour)

        # Iterate through each point in the contour
        for i in range(num_points):
            # Get current and next points cyclically
            current_point = contour[i]
            next_point = contour[(i + 1) % num_points]

            # Calculate the difference in x and y coordinates
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]

            # Determine the chain code based on the differences in coordinates
            if dx > 0 and dy == 0:  # East
                chain_code.append(0)
            elif dx > 0 and dy < 0:  # NorthEast
                chain_code.append(1)
            elif dx == 0 and dy < 0:  # North
                chain_code.append(2)
            elif dx < 0 and dy < 0:  # NorthWest
                chain_code.append(3)
            elif dx < 0 and dy == 0:  # West
                chain_code.append(4)
            elif dx < 0 and dy > 0:  # SouthWest
                chain_code.append(5)
            elif dx == 0 and dy > 0:  # South
                chain_code.append(6)
            elif dx > 0 and dy > 0:  # SouthEast
                chain_code.append(7)

        return chain_code

    def calculate_chain_code_4_connected(self, contour):
        """
        Calculates the 4-connected chain code for a given contour.

        Parameters:
        contour (list): List of (x, y) tuples representing the contour points.

        Returns:
        list: List of chain code numbers.

        Note:
        Assumes that the contour is closed (i.e., starts and ends at the same point).
        """
        chain_code = []
        num_points = len(contour)

        # Iterate through each point in the contour
        for i in range(num_points):
            # Get current and next points cyclically
            current_point = contour[i]
            next_point = contour[(i + 1) % num_points]

            # Calculate the difference in x and y coordinates
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]

            # Determine the chain code based on the differences in coordinates
            if dx > 0 and dy == 0:  # East
                chain_code.append(0)
            elif dx == 0 and dy < 0:  # North
                chain_code.append(1)
            elif dx < 0 and dy == 0:  # West
                chain_code.append(2)
            elif dx == 0 and dy > 0:  # South
                chain_code.append(3)

        return chain_code

    # --------------------------------Contour Perimeter--------------------------------

    def contour_perimeter(self, x_points, y_points):
        """
        Calculates the perimeter of a closed contour defined by (x, y) points.

        Parameters:
            x_points (list): List of x-coordinates of contour points.
            y_points (list): List of y-coordinates of contour points.

        Returns:
            float: The total perimeter distance.

        Note:
            Assumes that the contour is closed (i.e., starts and ends at the same point).
        """
        distance_sum = 0
        num_points = len(x_points)

        for i in range(num_points):
            next_point = (i + 1) % num_points  # Wrap around to the first point

            dx = x_points[next_point] - x_points[i]
            dy = y_points[next_point] - y_points[i]

            distance = (dx ** 2 + dy ** 2) ** 0.5  # Euclidean distance
            distance_sum += distance

        return distance_sum

    # --------------------------------Contour Area--------------------------------

    def contour_area(self, number_of_points, x_coordinates, y_coordinates):
        """
        Calculates the area of a simple polygon using the shoelace formula.

        Parameters:
            number_of_points (int): The total number of contour points.
            x_coordinates (list): List of x-coordinates of contour points.
            y_coordinates (list): List of y-coordinates of contour points.

        Returns:
            float: The positive absolute value of half the accumulated area.
        """
        area = 0.0
        j = number_of_points - 1

        # Calculate value of shoelace formula => 1/2 [ (x1y2 + x2y3 + … + xn-1yn + xny1) – (x2y1 + x3y2 + … + xnyn-1 + x1yn) ]
        for i in range(number_of_points):
            area += (x_coordinates[j] + x_coordinates[i]) * (y_coordinates[j] - y_coordinates[i])
            j = i  # j is the previous vertex to i

        return abs(area / 2.0)

    # --------------------------------Circle Detection --------------------------------
    def hough_circles(self, image, r_min: int = 20, r_max: int = 100, delta_r: int = 1,
                     num_thetas: int = 100, bin_threshold: float = 0.4, pixel_threshold: int = 20,
                     post_process: bool = True):
        """
        Performs circular Hough transform to detect circles in the given image.

        Args:
            image (numpy.ndarray): Input image (BGR format).
            r_min (int): Minimum radius of circles to be detected.
            r_max (int): Maximum radius of circles to be detected.
            delta_r (int): Step size for varying the radius.
            num_thetas (int): Number of angles to consider for each radius.
            bin_threshold (float): Threshold for considering a circle candidate as a detected circle.
            pixel_threshold (int): Threshold for post-processing to remove overlapping circles.
            post_process (bool): Flag to indicate whether to perform post-processing.

        Returns:
            numpy.ndarray: Output image with detected circles drawn.
        """
        # Convert the image to grayscale
        edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Canny edge detection to the grayscale image
        edge_image = cv2.Canny(edge_image, 100, 200)
        # Get the height and width of the edge image
        img_height, img_width = edge_image.shape[:2]
        # Calculate the angle step size based on the number of thetas
        dtheta = int(360 / num_thetas)
        # Generate an array of theta values evenly spaced from 0 to 360 degrees
        thetas = np.arange(0, 360, step=dtheta)
        # Generate an array of radius values within the specified range
        rs = np.arange(r_min, r_max, step=delta_r)
        # Generate a list of possible circle candidates based on theta and radius values
        circle_candidates = self.condidateCircles(thetas, rs, num_thetas)
        # Calculate the accumulator array for Hough transform
        accumulator = self.calculateAccumlator(img_height, img_width, edge_image, circle_candidates)
        # Create a copy of the original image
        output_img = image.copy()
        # List to store detected circles
        out_circles = []
        # Iterate through accumulator items sorted by vote count
        for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
            x, y, r = candidate_circle
            # Calculate the percentage of votes for the current circle candidate
            current_vote_percentage = votes / num_thetas
            # Check if the vote percentage exceeds the bin threshold
            if current_vote_percentage >= bin_threshold:
                # Add the circle candidate to the list of detected circles
                out_circles.append((x, y, r, current_vote_percentage))
                # Print the detected circle
                print("outcircle", out_circles)
        # Optionally post-process the detected circles to remove noise
        if post_process:
            out_circles = self.postProcces(out_circles, pixel_threshold)
        # Print the number of candidate circles detected
        print("Number of candidate circles:", len(out_circles))
        # Draw detected circles on the output image
        for x, y, r, v in out_circles:
            output_img = cv2.circle(output_img, (x, y), r, (0, 0, 255), thickness=2)
        # Return the output image with detected circles
        return output_img

    def calculateAccumlator(self, img_height, img_width, edge_image, circle_candidates):
        """
        Calculates the accumulator array for Hough transform by iterating through edge pixels and circle candidates,
        and accumulating votes for potential circle centers and radii.
        Args:
            img_height (int): Height of the edge image.
            img_width (int): Width of the edge image.
            edge_image (numpy.ndarray): Binary image containing edge pixels.
            circle_candidates (list): List of potential circle candidates represented as (radius, x_cos_theta, y_sin_theta) tuples.
        Returns:
            dict: Accumulator dictionary with (x_center, y_center, radius) tuples as keys and their corresponding votes as values.
        """

        # Print the number of edge pixels in the edge image
        print("Number of edge pixels:", np.count_nonzero(edge_image))
        # Print the number of circle candidates
        print("Number of circle candidates:", len(circle_candidates))
        # Initialize an accumulator dictionary to store votes for circle candidates
        accumulator = defaultdict(int)
        # Iterate through each pixel in the edge image
        for y in range(img_height):
            for x in range(img_width):
                # Check if the pixel is an edge pixel
                if edge_image[y][x] != 0:
                    # Iterate through each circle candidate
                    for r, rcos_t, rsin_t in circle_candidates:
                        # Calculate the center of the candidate circle
                        x_center = x - rcos_t
                        y_center = y - rsin_t
                        # Vote for the current candidate by incrementing its accumulator value
                        accumulator[(x_center, y_center, r)] += 1
        return accumulator

    def condidateCircles(self, thetas, rs, num_thetas):
        """
            Generates a list of circle candidates based on the given theta and radius values.
            Args:
                thetas (numpy.ndarray): Array of theta values in degrees.
                rs (numpy.ndarray): Array of radius values.
                num_thetas (int): Number of angles per theta.
            Returns:
                list: List of circle candidates represented as (radius, x_cos_theta, y_sin_theta) tuples.
            """
        # Print the number of theta values
        print("Number of thetas:", len(thetas))
        # Print the number of radius values
        print("Number of radii:", len(rs))
        # Print the number of angles per theta
        print("Number of angles per theta:", num_thetas)
        # Calculate cosine and sine values of theta
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))
        # Initialize an empty list to store circle candidates
        circle_candidates = []
        # Iterate through each radius value
        for r in rs:
            # Iterate through each theta value
            for t in range(num_thetas):
                # Calculate the coordinates of the candidate circle's center
                circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
        # Print the generated circle candidates
        print("circle_candyyyy", circle_candidates)
        return circle_candidates

    def postProcces(self, output_circle, pixel_threshold):
        """
        Performs post-processing on detected circles to remove duplicates or closely overlapping circles.
        Args:
            output_circle (list): List of detected circles represented as (x_center, y_center, radius, vote_percentage) tuples.
            pixel_threshold (int): Threshold for considering two circles as overlapping.
        Returns:
            list: List of post-processed circles after removing duplicates or closely overlapping circles.
        """
        # Initialize a list to store post-processed circles
        postprocess_circles = []
        # Iterate through each detected circle
        for x, y, r, v in output_circle:
            # Check if the circle is close to an existing circle
            is_close_to_existing_circle = False
            for xc, yc, rc, v_inner in postprocess_circles:
                if abs(x - xc) <= pixel_threshold or abs(y - yc) <= pixel_threshold or abs(
                        r - rc) <= pixel_threshold:
                    is_close_to_existing_circle = True
                    break
                    # Add the circle to the list if it's not close to any existing circle
            if not is_close_to_existing_circle:
                postprocess_circles.append((x, y, r, v))
        return postprocess_circles

    def apply_circle_detection(self):
        # Get the parameter values from UI sliders
        min_radius = self.horizontalSlider_2.value()
        max_radius = self.horizontalSlider_3.value()
        bin_threshold = self.horizontalSlider_4.value() / 10.0  # Divide by 100 to get value between 0 and 1
        pixel_threshold = self.horizontalSlider_5.value()
        # Get the current original image
        curr_image = self.get_current_orignal_img()
        # Perform circle detection if there is an image
        if curr_image is not None:
            image_circle = self.houghCircles(self.image, r_min=min_radius, r_max=max_radius,
                                             bin_threshold=bin_threshold, pixel_threshold=pixel_threshold)
            # Display the filtered image
            self.display_filtered_img(image_circle, self.output_image, self.output_scene)

    def set_sliders(self, slider):
        slider.setMinimum(1)
        slider.setMaximum(100)



def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()