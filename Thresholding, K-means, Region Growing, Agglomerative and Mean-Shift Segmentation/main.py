import random
import numpy as np
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QPainter, QPen
from PyQt5.QtWidgets import *
import sys
import cv2
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree, cKDTree
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import timeit

from sklearn.neighbors import BallTree





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
        self.tabWidget = self.findChild(QTabWidget, 'tabWidget')
        self.browseButton_1.clicked.connect(lambda: self.browse_image(self.input_scene_1))  # Connect the clicked signal to browse_image function
        self.browseButton_2.clicked.connect(lambda: self.browse_image(self.input_scene_2))  # Connect the clicked signal to browse_image function
        self.browseButton_3.clicked.connect(lambda: self.browse_image(self.input_scene_3))  # Connect the clicked signal to browse_image function
        # Connect the currentChanged signal of the tabWidget to the handle_sliders function
        self.tabWidget.currentChanged.connect(self.handle_sliders)
        # Connect the currentIndexChanged signal of the segmentationCombox to the on_combobox_changed function
        self.segmentationCombox.currentIndexChanged.connect(self.handle_sliders)
        self.horizontalSlider.valueChanged.connect(lambda value: self.slider_value_changed(self.horizontalSlider, self.label_4, value))
        self.horizontalSlider_2.valueChanged.connect(lambda value: self.slider_value_changed(self.horizontalSlider_2, self.label_3, value))
        self.Apply_pushButton.clicked.connect(self.handle_apply_button)
        self.Apply_pushButton_2.clicked.connect(self.handle_apply_button)
        self.Apply_pushButton.clicked.connect(self.handle_apply_button)
        ######################################################################################################
        self.apply_btn.clicked.connect(self.handle_threshold_apply_btn)
        self.select_block_size_label.setVisible(False)
        self.block_size_slider.setVisible(False)
        self.block_size_val_label.setVisible(False)
        self.radioButton.clicked.connect(self.show_slider)
        self.radioButton_2.clicked.connect(self.hide_slider)

        self.block_size_slider.valueChanged.connect(self.block_size_slider_value_changed)

        self.number_of_classes_slider.valueChanged.connect(self.number_of_classes_slider_value_changed)


        self.block_size=0


    def block_size_slider_value_changed(self):
        value = self.block_size_slider.value()
        self.block_size = value
        self.block_size_val_label.setText(f"val={str(value)}")
    def number_of_classes_slider_value_changed(self):
        value = self.number_of_classes_slider.value()//10
        self.number_of_classes = value
        self.number_of_classes_val_label.setText(f"number of classes={str(value)}")
    def show_slider(self):

        self.select_block_size_label.setVisible(True)
        self.block_size_slider.setVisible(True)
        self.block_size_val_label.setVisible(True)
    def hide_slider(self):
        self.select_block_size_label.setVisible(False)
        self.block_size_slider.setVisible(False)
        self.block_size_val_label.setVisible(False)

    def show_classes_slider(self):
        self.select_number_of_classes_label.setVisible(True)
        self.number_of_classes_slider.setVisible(True)
        self.number_of_classes_val_label.setVisible(True)
    def hide_classes_slider(self):
        self.select_number_of_classes_label.setVisible(False)
        self.number_of_classes_slider.setVisible(False)
        self.number_of_classes_val_label.setVisible(False)
    

    def handle_threshold_apply_btn(self):
        selected_item = self.thresholdCombox.currentText()

        if selected_item == "Optimal Thresholding":
            if self.radioButton.isChecked()==1:
                self.apply_local_optimal_threshold(self.block_size)
            if self.radioButton_2.isChecked()==1:
                self.apply_global_optimal_threshold()


        if selected_item == "Otsu Thresholding":
            if self.radioButton_2.isChecked() == 1:
                self.global_otsu_thresholding(self.input_data)

            if self.radioButton.isChecked() == 1:
                self.local_otsu_thresholding(self.input_data, self.block_size)
        
        if selected_item == "Spectral Thresholding":
            self.radioButton.clicked.connect(self.show_classes_slider)
            self.radioButton_2.clicked.connect(self.show_classes_slider)
            if self.radioButton_2.isChecked()==1:
                self.spectral_thresholding(self.input_data, self.number_of_classes)
            if self.radioButton.isChecked() == 1:
                self.local_spectral_thresholding(self.input_data, self.block_size, self.number_of_classes)
        

            

    def apply_local_optimal_threshold(self, block_size):
        print(f"block size val = {block_size}")

        """
        Apply local optimal thresholding to the input image.

        Parameters:
            block_size (int): Size of the square blocks for local thresholding.

        Returns:
            None
        """


        image = cv2.cvtColor(self.input_data, cv2.COLOR_BGR2GRAY)
        if image.shape[0] != image.shape[1]:
            if image.shape[0] > image.shape[1]:
                resizedImage = cv2.resize(image, (image.shape[0], image.shape[0]))
            else:
                resizedImage = cv2.resize(image, (image.shape[1], image.shape[1]))
        else:
            resizedImage = image
        rows = resizedImage.shape[0]
        cols = resizedImage.shape[1]

        if block_size <= 2:
            print("Error in local thresholding , block size should be greater than 2 ! ")
            exit()

        if block_size > image.shape[0] and block_size > image.shape[1]:
            print("Error local thresholding , block size should be smaller than image size!")
            exit()

        outputImage = np.zeros(resizedImage.shape)

        for r in range(0, rows, block_size):
            for c in range(0, cols, block_size):
                # Extract blocks
                block = resizedImage[r:min(r + block_size, rows), c:min(c + block_size, cols)]
                # Get initial background mean (4 corners)
                background = [block[0, 0], block[0, block.shape[1] - 1], block[block.shape[0] - 1, 0],
                              block[block.shape[0] - 1, block.shape[1] - 1]]
                background_mean = np.mean(background)
                # Get initial foreground mean
                foreground_mean = np.mean(block) - background_mean
                # Get initial threshold
                thresh = (background_mean + foreground_mean) / 2.0
                while True:
                    old_thresh = thresh
                    new_foreground = block[np.where(block >= thresh)]
                    new_background = block[np.where(block < thresh)]
                    if new_background.size:
                        new_background_mean = np.mean(new_background)
                    else:
                        new_background_mean = 0
                    if new_foreground.size:
                        new_foreground_mean = np.mean(new_foreground)
                    else:
                        new_foreground_mean = 0
                    # Update threshold
                    thresh = (new_background_mean + new_foreground_mean) / 2
                    if old_thresh == thresh:
                        break

                # Convert to binary [ (0 , 255) only]
                thresholdedBlock = np.where(block >= thresh, 255, 0)

                # Fill the output image for each block
                outputImage[r:min(r + block_size, rows), c:min(c + block_size, cols)] = thresholdedBlock

        # Resize output image back to the original size
        outputImage = cv2.resize(outputImage, (image.shape[1], image.shape[0]))
        print(f"result shape = {outputImage.shape}")
        binary_img= outputImage
        # Save the plot as an image file
        plot_path = "binary_plot.png"
        plt.imshow(binary_img, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

        # Close the plot to avoid displaying it in a separate window
        plt.close()
        # Load the saved plot image
        binary_plot_img = cv2.imread(plot_path)

        if binary_plot_img is None:
            QMessageBox.critical(self.centralwidget, "Error", "Unable to load plot image.")
            return

        # Convert the BGR image to RGB
        binary_plot_img_rgb = cv2.cvtColor(binary_plot_img, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        plot_height, plot_width, _ = binary_plot_img_rgb.shape
        plot_bytes_per_line = 3 * plot_width
        plot_q_image = QImage(binary_plot_img_rgb.data, plot_width, plot_height, plot_bytes_per_line,
                              QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        plot_pixmap = QPixmap.fromImage(plot_q_image)

        # Clear the output scene
        self.output_scene_1.clear()

        # Add the pixmap to the output scene
        self.output_scene_1.addPixmap(plot_pixmap)

        # Set scene rect to match image dimensions
        self.output_scene_1.setSceneRect(0, 0, plot_width, plot_height)

        # Fit the view
        self.output_graphicsView_1.fitInView(self.output_scene_1.sceneRect(), Qt.KeepAspectRatio)
        # Delete the plot image file
        os.remove(plot_path)



    def apply_global_optimal_threshold(self):
        """
            Apply global optimal thresholding to the input image.

            Returns:
                None
            """
        print("apply_global_optimal_threshold ")
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(self.input_data, cv2.COLOR_BGR2GRAY)

        threshold_value = self.Optimal(gray_img)
        print("optimal done")
        binary_img = (gray_img < threshold_value).astype(np.uint8) * 255

        # Convert the binary image to QImage
        q_binary_image = QImage(binary_img.data, binary_img.shape[1], binary_img.shape[0], binary_img.shape[1],
                                QImage.Format_Grayscale8)

        # Create a QPixmap from the QImage
        pixmap_binary = QPixmap.fromImage(q_binary_image)

        # Clear the output scene
        self.output_scene_1.clear()

        # Add the binary pixmap to the output scene
        self.output_scene_1.addPixmap(pixmap_binary)

        # Set scene rect to match image dimensions
        self.output_scene_1.setSceneRect(0, 0, binary_img.shape[1], binary_img.shape[0])

        # Fit the view to display the entire image while preserving aspect ratio
        self.output_graphicsView_1.fitInView(self.output_scene_1.sceneRect(), Qt.KeepAspectRatio)

    def Optimal(self, img):
        """
            Calculate the optimal threshold value for global thresholding using Otsu's method.

            Parameters:
                img (numpy.ndarray): Input grayscale image.

            Returns:
                float: Optimal threshold value.
            """
        print("Optimal entered")
        background_sum = (img[0, 0] + img[0, -1] + img[-1, 0] + img[-1, -1])
        foreground_sum = np.sum(img) - background_sum
        background_mean = background_sum / 4
        foreground_mean = foreground_sum / (np.size(img) - 4)
        t = (foreground_mean + background_mean) / 2
        while True:
            background_mean = np.mean(img[img < t])
            foreground_mean = np.mean(img[img > t])

            if (t == (background_mean + foreground_mean) / 2):
                break
            t = (background_mean + foreground_mean) / 2
        return t

    def global_threshold(self, image, threshold):
        """
            Binarize the input image using a global threshold.

            Parameters:
                image (numpy.ndarray): A NumPy array representing the input image.
                threshold (int): The threshold value for binarization.

            Returns:
                numpy.ndarray: A binary image where pixels above the threshold are set to 256, and others to 0.
            """
        binary = image > threshold
        for i in range(0, binary.shape[0], 1):
            for j in range(0, (binary.shape[1]), 1):
                if binary[i][j] == True:
                    binary[i][j] = 256
                else:
                    binary[i][j] = 0
        return binary

    def global_otsu_thresholding(self, image):
        """
            Apply global Otsu thresholding to the input image.

            Parameters:
                image (numpy.ndarray): A NumPy array representing the input image.

            Returns:
                None
            """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        no_rows = image.shape[0]
        no_cols = image.shape[1]
        imageSize = no_rows * no_cols
        graylevel = range(0, 256)
        ### Histogram
        hist = [0] * 256
        for i in range(0, 256):
            hist[i] = len(np.extract(np.asarray(image) == graylevel[i], image))
        # counts,histo = np.histogram(image)
        variance = []
        for i in range(256):
            threshold = i
            background_gray_level = np.extract(np.asarray(graylevel) < threshold, graylevel)
            foreground_gray_level = np.extract(np.asarray(graylevel) >= threshold, graylevel)
            background_hist = []
            foreground_hist = []

            ##### Weights(W_g, W_f)
            back_weight = 0
            fore_weight = 0
            ##### mean (m_g, m_f)
            back_mean = 0
            fore_mean = 0

            background_length = len(background_gray_level)
            foreground_length = len(foreground_gray_level)

            if background_length != 0:
                for i in background_gray_level:
                    background_hist.append(hist[i])
                    total_back_hist = sum(background_hist)
                    back_weight = float(total_back_hist) / imageSize

                if back_weight != 0:
                    back_mean = np.sum(np.multiply(background_gray_level, np.asarray(background_hist))) / float(
                        sum(background_hist))

            if foreground_length != 0:
                for i in foreground_gray_level:
                    foreground_hist.append(hist[i])
                    total_fore_hist = sum(foreground_hist)
                    fore_weight = float(total_fore_hist) / imageSize

                if fore_weight != 0:
                    fore_mean = np.sum(np.multiply(foreground_gray_level, np.asarray(foreground_hist))) / float(
                        sum(foreground_hist))

            variance.append(back_weight * fore_weight * ((back_mean - fore_mean) ** 2))

        max_variance = np.max(variance)
        Threshold = variance.index(max_variance)
        outputImage = image.copy()
        print(Threshold)
        outputImage = self.global_threshold(image, Threshold)
        final_img = outputImage

        # Save the plot as an image file
        plot_path = "binary_plot.png"
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

        # Close the plot to avoid displaying it in a separate window
        plt.close()
        # Load the saved plot image
        binary_plot_img = cv2.imread(plot_path)

        if binary_plot_img is None:
            QMessageBox.critical(self.centralwidget, "Error", "Unable to load plot image.")
            return

        # Convert the BGR image to RGB
        binary_plot_img_rgb = cv2.cvtColor(binary_plot_img, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        plot_height, plot_width, _ = binary_plot_img_rgb.shape
        plot_bytes_per_line = 3 * plot_width
        plot_q_image = QImage(binary_plot_img_rgb.data, plot_width, plot_height, plot_bytes_per_line,
                              QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        plot_pixmap = QPixmap.fromImage(plot_q_image)

        # Clear the output scene
        self.output_scene_1.clear()

        # Add the pixmap to the output scene
        self.output_scene_1.addPixmap(plot_pixmap)

        # Set scene rect to match image dimensions
        self.output_scene_1.setSceneRect(0, 0, plot_width, plot_height)

        # Fit the view
        self.output_graphicsView_1.fitInView(self.output_scene_1.sceneRect(), Qt.KeepAspectRatio)
        # Delete the plot image file
        os.remove(plot_path)

    def local_otsu_thresholding(self, image, block_size):
        """
            Apply local Otsu thresholding to the input image.

            Parameters:
                image (numpy.ndarray): A NumPy array representing the input image.
                block_size (int): The size of square blocks for local thresholding.

            Returns:
                None
            """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ## Checking the dimension of image comparing to block size
        # resize the input image
        if image.shape[0] < image.shape[1]:
            resized_image = cv2.resize(image, (image.shape[1], image.shape[1]))
        else:
            resized_image = cv2.resize(image, (image.shape[0], image.shape[0]))

        no_rows = resized_image.shape[0]
        no_cols = resized_image.shape[1]

        if block_size > resized_image.shape[0] and block_size > resized_image.shape[1]:
            print("You can not apply local thresold in image")
            return 0

        output_image = resized_image.copy()

        #### Then apply the otsu algorithm
        ### The difference between local and global is we divide the image into windows
        for r in range(0, resized_image.shape[0], block_size):
            for c in range(0, resized_image.shape[1], block_size):
                #### Blocks
                block = resized_image[r:min(r + block_size, no_rows), c:min(c + block_size, no_cols)]
                size_block = np.size(block)

                graylevel = range(0, 256)
                ### Histogram
                hist = [0] * 256
                for i in range(0, 256):
                    hist[i] = len(np.extract(np.asarray(block) == graylevel[i], block))

                variance = []
                for i in range(256):
                    threshold = i
                    background_gray_level = np.extract(np.asarray(graylevel) < threshold, graylevel)
                    foreground_gray_level = np.extract(np.asarray(graylevel) >= threshold, graylevel)
                    background_hist = []
                    foreground_hist = []

                    ##### Weights(W_g, W_f)
                    back_weight = 0
                    fore_weight = 0
                    ##### mean (m_g, m_f)
                    back_mean = 0
                    fore_mean = 0

                    background_length = len(background_gray_level)
                    foreground_length = len(foreground_gray_level)

                    if background_length != 0:
                        for i in background_gray_level:
                            background_hist.append(hist[i])
                            total_back_hist = sum(background_hist)
                            back_weight = float(total_back_hist) / size_block

                        if back_weight != 0:
                            back_mean = np.sum(np.multiply(background_gray_level, np.asarray(background_hist))) / float(
                                sum(background_hist))

                    if foreground_length != 0:
                        for i in foreground_gray_level:
                            foreground_hist.append(hist[i])
                            total_fore_hist = sum(foreground_hist)
                            fore_weight = float(total_fore_hist) / size_block

                        if fore_weight != 0:
                            fore_mean = np.sum(np.multiply(foreground_gray_level, np.asarray(foreground_hist))) / float(
                                sum(foreground_hist))

                    variance.append(back_weight * fore_weight * ((back_mean - fore_mean) ** 2))

                max_variance = np.max(variance)
                Threshold = variance.index(max_variance)

                thresholded_block = self.global_threshold(block, Threshold)

                output_image[r:min(r + block_size, no_rows), c:min(c + block_size, no_cols)] = thresholded_block

        output_image = cv2.resize(output_image, (image.shape[0], image.shape[1]))
        final_img =  output_image

        # Save the plot as an image file
        plot_path = "binary_plot.png"
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

        # Close the plot to avoid displaying it in a separate window
        plt.close()
        # Load the saved plot image
        binary_plot_img = cv2.imread(plot_path)

        if binary_plot_img is None:
            QMessageBox.critical(self.centralwidget, "Error", "Unable to load plot image.")
            return

        # Convert the BGR image to RGB
        binary_plot_img_rgb = cv2.cvtColor(binary_plot_img, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        plot_height, plot_width, _ = binary_plot_img_rgb.shape
        plot_bytes_per_line = 3 * plot_width
        plot_q_image = QImage(binary_plot_img_rgb.data, plot_width, plot_height, plot_bytes_per_line,
                              QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        plot_pixmap = QPixmap.fromImage(plot_q_image)

        # Clear the output scene
        self.output_scene_1.clear()

        # Add the pixmap to the output scene
        self.output_scene_1.addPixmap(plot_pixmap)

        # Set scene rect to match image dimensions
        self.output_scene_1.setSceneRect(0, 0, plot_width, plot_height)

        # Fit the view
        self.output_graphicsView_1.fitInView(self.output_scene_1.sceneRect(), Qt.KeepAspectRatio)
        # Delete the plot image file
        os.remove(plot_path)
    ###################################################################################################################
    
    def histogram(self,image, bins=256):
        """
        Compute the histogram of the input image.

        Args:
        - image: Input image (grayscale).
        - bins: Number of bins for the histogram.

        Returns:
        - hist: Computed histogram.
        """
        hist, _ = np.histogram(image, bins=bins, range=(0, 256))
        return hist

    def normalized_histogram(self,image, bins=256):
        """
        Compute the normalized histogram of the input image.

        Args:
        - image: Input image (grayscale).
        - bins: Number of bins for the histogram.

        Returns:
        - hist_norm: Computed normalized histogram.
        """
        hist = self.histogram(image, bins)
        hist_norm = hist.astype(np.float32) / np.sum(hist)
        return hist_norm

    def otsu_threshold(self,hist_norm):
        """
        Compute the Otsu threshold for a given normalized histogram.

        Args:
        - hist_norm: Normalized histogram.

        Returns:
        - threshold: Otsu threshold.
        """
        threshold = 0
        max_variance = 0

        for t in range(1, len(hist_norm)):
            w0 = np.sum(hist_norm[:t])
            w1 = np.sum(hist_norm[t:])

            if w0 == 0 or w1 == 0:
                continue

            mu0 = np.sum(np.arange(t) * hist_norm[:t]) / w0
            mu1 = np.sum(np.arange(t, len(hist_norm)) * hist_norm[t:]) / w1

            variance = w0 * w1 * ((mu0 - mu1) ** 2)
            if variance > max_variance:
                max_variance = variance
                threshold = t

        return threshold

    def multi_otsu_threshold(self,hist_norm, classes):
        """
        Compute multiple Otsu thresholds for a given normalized histogram.

        Args:
        - hist_norm: Normalized histogram.
        - classes: Number of classes.

        Returns:
        - thresholds: List of Otsu thresholds.
        """
        thresholds = []
        hist_size = len(hist_norm)
        total_bins = hist_size - 1

        if classes < 2 or classes > total_bins:
            raise ValueError("Number of classes must be between 2 and total number of bins - 1.")

        for _ in range(classes - 1):
            threshold = self.otsu_threshold(hist_norm)
            thresholds.append(threshold)

            # Update histogram for the next iteration
            hist_norm = hist_norm[threshold:]

        return thresholds

    def apply_threshold(self,image, thresholds):
        """
        Apply thresholding on the input image using the provided thresholds.

        Args:
        - image: Input image (grayscale).
        - thresholds: List of threshold values.

        Returns:
        - segmented_image: Segmented image after thresholding.
        """
        segmented_image = np.zeros_like(image)
        for threshold in thresholds:
            segmented_image[image >= threshold] += 1
        return segmented_image

    # Example usage:
    def spectral_thresholding(self,image,num_classes):
        # Load your image (replace 'image_path' with your image file path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        # Compute normalized histogram
        hist_norm = self.normalized_histogram(image)

        # # Choose the number of classes (thresholds + 1)
        # num_classes = 4

        # Compute multiple Otsu thresholds
        thresholds = self.multi_otsu_threshold(hist_norm, num_classes)

        # Apply thresholding
        segmented_image = self.apply_threshold(image, thresholds)

        # Save the plot as an image file
        plot_path = "binary_plot.png"
        plt.imshow(segmented_image, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

        # Close the plot to avoid displaying it in a separate window
        plt.close()
        # Load the saved plot image
        binary_plot_img = cv2.imread(plot_path)

        if binary_plot_img is None:
            QMessageBox.critical(self.centralwidget, "Error", "Unable to load plot image.")
            return

        # Convert the BGR image to RGB
        binary_plot_img_rgb = cv2.cvtColor(binary_plot_img, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        plot_height, plot_width, _ = binary_plot_img_rgb.shape
        plot_bytes_per_line = 3 * plot_width
        plot_q_image = QImage(binary_plot_img_rgb.data, plot_width, plot_height, plot_bytes_per_line,
                              QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        plot_pixmap = QPixmap.fromImage(plot_q_image)

        # Clear the output scene
        self.output_scene_1.clear()

        # Add the pixmap to the output scene
        self.output_scene_1.addPixmap(plot_pixmap)

        # Set scene rect to match image dimensions
        self.output_scene_1.setSceneRect(0, 0, plot_width, plot_height)

        # Fit the view
        self.output_graphicsView_1.fitInView(self.output_scene_1.sceneRect(), Qt.KeepAspectRatio)
        # Delete the plot image file
        os.remove(plot_path)
 ############################################################################################################
    def local_otsu_threshold(self,image, num_classes):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).reshape(-1)
        hist_norm = hist.astype(float) / hist.sum()
        intensity_values = np.arange(256)
        mu = np.dot(intensity_values, hist_norm)
        thresholds = np.arange(1, num_classes)
        p = np.cumsum(hist_norm)
        mu_t = np.cumsum(intensity_values * hist_norm)
        sigma_b_squared = (mu_t[-1] * p - mu_t) ** 2 / (p * (1 - p))
        max_idx = np.argmax(sigma_b_squared)
        threshold_values = [np.sum(hist_norm[:max_idx + 1] * intensity_values[:max_idx + 1]) / np.sum(hist_norm[:max_idx + 1])]
        threshold_values += [np.sum(hist_norm[max_idx + 1:] * intensity_values[max_idx + 1:]) / np.sum(hist_norm[max_idx + 1:])]
        return threshold_values

    def local_multi_otsu_threshold(self,image, num_thresholds):
        thresholds = [0] + self.local_otsu_threshold(image, num_thresholds - 1) + [255]
        return thresholds

    def local_threshold(self,image, window_size, num_thresholds):
        rows, cols = image.shape
        thresholded_image = np.zeros_like(image)
        for i in range(rows - window_size + 1):
            for j in range(cols - window_size + 1):
                window = image[i:i+window_size, j:j+window_size]
                local_thresholds = self.local_multi_otsu_threshold(window, num_thresholds)
                for k in range(len(local_thresholds) - 1):
                    mask = (window >= local_thresholds[k]) & (window < local_thresholds[k+1])
                    thresholded_image[i:i+window_size, j:j+window_size][mask] = (255 / (num_thresholds - 1)) * k
        return thresholded_image
    def local_spectral_thresholding(self,image,window_size,num_thresholds):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholded_image = self.local_threshold(image, window_size, num_thresholds)
        # Save the plot as an image file
        plot_path = "binary_plot.png"
        plt.imshow(thresholded_image, cmap='gray')
        plt.axis('off')  # Hide axis
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)

        # Close the plot to avoid displaying it in a separate window
        plt.close()
        # Load the saved plot image
        binary_plot_img = cv2.imread(plot_path)

        if binary_plot_img is None:
            QMessageBox.critical(self.centralwidget, "Error", "Unable to load plot image.")
            return

        # Convert the BGR image to RGB
        binary_plot_img_rgb = cv2.cvtColor(binary_plot_img, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        plot_height, plot_width, _ = binary_plot_img_rgb.shape
        plot_bytes_per_line = 3 * plot_width
        plot_q_image = QImage(binary_plot_img_rgb.data, plot_width, plot_height, plot_bytes_per_line,
                              QImage.Format_RGB888)

        # Create a QPixmap from the QImage
        plot_pixmap = QPixmap.fromImage(plot_q_image)

        # Clear the output scene
        self.output_scene_1.clear()

        # Add the pixmap to the output scene
        self.output_scene_1.addPixmap(plot_pixmap)

        # Set scene rect to match image dimensions
        self.output_scene_1.setSceneRect(0, 0, plot_width, plot_height)

        # Fit the view
        self.output_graphicsView_1.fitInView(self.output_scene_1.sceneRect(), Qt.KeepAspectRatio)
        # Delete the plot image file
        os.remove(plot_path)

    ###############################################################################################################


    def handle_ui(self):
        # Create a QGraphicsScene
        self.input_scene_1 = QtWidgets.QGraphicsScene()
        self.input_graphicsView_1.setScene(self.input_scene_1)
        self.input_scene_2 = QtWidgets.QGraphicsScene()
        self.input_graphicsView_2.setScene(self.input_scene_2)
        self.input_scene_3 = QtWidgets.QGraphicsScene()
        self.input_graphicsView_3.setScene(self.input_scene_3)
        self.output_scene_1 = QtWidgets.QGraphicsScene()
        self.output_graphicsView_1.setScene(self.output_scene_1)
        self.output_scene_2 = QtWidgets.QGraphicsScene()
        self.output_graphicsView_2.setScene(self.output_scene_2)
        self.output_scene_3 = QtWidgets.QGraphicsScene()
        self.output_graphicsView_3.setScene(self.output_scene_3)

        # Connect mouse press event to handle_region_growing_click method
        self.output_graphicsView_2.mousePressEvent = self.handle_region_growing_click

    def browse_image(self, scene):
        # Open a file dialog to select an image file
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.webp *.png *.jpeg *.avif)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]

            # Load the selected image file using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Convert the BGR image to RGB
            self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.current_img = image

            # Convert the image to QImage
            height, width, _ = self.image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.input_data = self.image_rgb
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
            # Connect mouse press event to handle_seed_point method

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

    def handle_sliders(self, index):
        # Check if the second tab is selected
        if self.tabWidget.currentIndex() == 1:
            for slider in [self.horizontalSlider, self.horizontalSlider_2]:
                slider.setValue(0)
            # Get the currently selected item in the segmentationCombox combobox
            selected_item = self.segmentationCombox.currentText()

            if selected_item == "K-means Segmentation":
                self.horizontalSlider.setMaximum(100)
                self.horizontalSlider_2.show()
                self.label_2.show()
                self.label_3.show()
                self.label.setText("# Clusters")
                self.label_2.setText("# Iterations")
            elif selected_item == "Mean-Shift Segmentation":
                self.horizontalSlider.setMaximum(500)
                self.label.setText("Bandwidth")
                self.horizontalSlider_2.hide()
                self.label_2.hide()
                self.label_3.hide()
            elif selected_item == "Region Growing":
                self.horizontalSlider.setMaximum(100)
                self.label.setText("Threshold")
                self.label_2.hide()
                self.label_3.hide()
                self.horizontalSlider_2.hide()
            elif selected_item == "Agglomerative":
                self.horizontalSlider_2.show()
                self.horizontalSlider.show()
                self.label_2.show()
                self.label.setText("Clusters")
                self.label_2.setText("Threshold")

    def slider_value_changed(self, slider, label, value):
        # Check if the value is from the correct slider
        if slider.value() == value:
            label.setText(str(value))

    def handle_apply_button(self):
        if self.tabWidget.currentIndex() == 2:
            self.rgb_to_luv()
        else:
            # Get the currently selected item in the segmentationCombox combobox
            selected_item = self.segmentationCombox.currentText()

            if selected_item == "K-means Segmentation":
                self.kmeans_segmentation()
            elif selected_item == "Mean-Shift Segmentation":
                self.threshold = 60
                self.clusters = []
                self.bandwidth = self.horizontalSlider.value()
                self.mean_shift_segmentation(self.current_img)
            elif selected_item == "Region Growing":
                dummy_event = QMouseEvent(QEvent.MouseButtonPress, QPointF(0, 0), Qt.LeftButton, Qt.LeftButton,
                                        Qt.NoModifier)
                self.handle_region_growing_click(dummy_event)

            elif selected_item == "Agglomerative":
                # Calculate function run time
                start_time = timeit.default_timer()
                # Number of clusters
                k = self.horizontalSlider.value()
                initial_clusters = self.horizontalSlider_2.value()
                source = np.copy(self.current_img)
                segmented_image = self.apply_agglomerative(source=source, clusters_numbers=k, initial_clusters=initial_clusters)
                self.display_filtered_img(segmented_image, self.output_graphicsView_2, self.output_scene_2)
                # Function end
                end_time = timeit.default_timer()
                # Show only 5 digits after floating point
                elapsed_time = format(end_time - start_time, '.5f')
                print("elapsed_time: ", elapsed_time)

    def kmeans_segmentation(self):
        """
        This function applies K-means clustering to segment an image into different regions.

        The function works as follows:
        1. Reshape the image to a 2D array of pixels and convert the pixel values to floating point.
        2. Get the number of clusters and maximum iterations from the user interface.
        3. Initialize random centers.
        4. For each iteration, compute distances from pixels to centers and assign each pixel to the closest center.
        5. Compute new centers as the mean of the assigned pixels. If a cluster has no points, reinitialize its center.
        6. Convert the cluster centers back to 8 bit values.
        7. Map the labels to the cluster centers to create the segmented image.
        8. Reshape the segmented image back to the original image shape and display it.
        """

        # Get the current image
        image = self.current_img

        # Reshape the image to a 2D array of pixels
        pixels = image.reshape((-1, 3))

        # Convert the pixel values to floating point for more precise computations
        pixels = np.float32(pixels)

        # Get the number of clusters and maximum iterations from the user interface
        k = self.horizontalSlider.value()
        iterations = self.horizontalSlider_2.value()

        # Initialize random centers
        centers = pixels[np.random.choice(pixels.shape[0], size=k, replace=False)]

        for _ in range(iterations):
            # Compute distances from pixels to centers
            distances = np.linalg.norm(pixels[:, None] - centers, axis=2)

            # Assign each pixel to the closest center
            labels = np.argmin(distances, axis=1)

            # Compute new centers as the mean of the assigned pixels
            for i in range(k):
                if np.sum(labels == i) > 0:
                    # If a cluster has points, compute its new center
                    centers[i] = np.mean(pixels[labels == i], axis=0)
                else:
                    # If a cluster has no points, reinitialize its center
                    centers[i] = pixels[np.random.choice(pixels.shape[0])]

        # Convert the cluster centers back to 8 bit values
        centers = np.uint8(centers)

        # Map the labels to the cluster centers to create the segmented image
        segmented_image = centers[labels.flatten()]

        # Reshape the segmented image back to the original image shape
        segmented_image = segmented_image.reshape(image.shape)

        cv2.imwrite('k-means.png', segmented_image)

        # Display the segmented image
        self.display_filtered_img(segmented_image, self.output_graphicsView_2, self.output_scene_2)

        # Compute and print the number of unique colors in the segmented image
        colors = np.unique(segmented_image.reshape(-1, segmented_image.shape[2]), axis=0)
        print(f"# colors in the segmented image is {len(colors)}")


    def distance(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.sqrt(np.sum((a - b) ** 2))

    def get_points_within_bandwidth(self, center, bandwidth, points):
        return [p for p in points if self.distance(p, center) <= bandwidth]

    def region_growing(self,image, seed_point, threshold):
        """
            Perform region growing segmentation on the input image starting from the given seed point.

            Parameters:
                image (numpy.ndarray): The input image to be segmented.
                seed_point (tuple): The seed point (x, y) from which the region growing process will start.
                threshold (float): The threshold value specifying the maximum allowable difference 
                                  in intensity between the seed pixel and its neighbors for inclusion in the region.

            Returns:
                numpy.ndarray: The segmented image where pixels belonging to the region are set to 255,
                               and all other pixels are set to 0.
        """     
        # Initialize output image
        segmented_image = np.zeros(image.shape[:2], dtype=np.uint8)
        # Get image dimensions
        height, width = image.shape[:2]
        # Initialize a queue for pixel traversal
        queue = []
        # Add seed point to the queue
        queue.append(seed_point)
        # Get the color value of the seed point
        seed_value = image[seed_point[1], seed_point[0]]
        print("seed_value",seed_value)
        # Define 4-connected neighborhood
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Iterate until the queue is empty
        while queue:
            # Pop the front element from the queue
            current_point = queue.pop(0)
            # Set the current pixel in the segmented image
            segmented_image[current_point[1], current_point[0]] = 255

            # Check the 4-connected neighbors of the current pixel
            for neighbor in neighbors:
                # Calculate coordinates of the neighbor
                x = current_point[0] + neighbor[0]  #this calculate nuber if neghbourpixel x = 3 + (-1) = 2

                y = current_point[1] + neighbor[1]

                # Check if the neighbor is within image boundaries
                if x >= 0 and y >= 0 and x < width and y < height:
                    # Check if the neighbor pixel is not yet segmented
                    if segmented_image[y, x] == 0:
                        # Get the color value of the neighbor pixel
                        neighbor_value = image[y, x]
                        # Check if the color difference is less than the threshold
                        if np.linalg.norm(seed_value - neighbor_value) < threshold:
                            # Add the neighbor pixel to the queue
                            queue.append((x, y))
                            # Mark the neighbor pixel as segmented
                            segmented_image[y, x] = 255
        output_image = image.copy()
        output_image[segmented_image == 255] = [0, 255, 255]
        self.display_filtered_img(output_image, self.output_graphicsView_2, self.output_scene_2)


    def handle_region_growing_click(self, event):
        print("Output image clicked!")  # Check if the method is called

        # Convert mouse position to scene coordinates
        mouse_position = event.pos()
        scene_pos = self.output_graphicsView_2.mapToScene(mouse_position)
        # Extract seed point coordinates
        seed_point = (int(scene_pos.x()), int(scene_pos.y()))

        # Perform region growing using the extracted seed point
        self.region_growing(self.current_img, seed_point, self.horizontalSlider.value())

    def rgb_to_luv(self):
        """
        This function converts an RGB image to LUV color space.

        The function first scales the RGB values to the range [0, 1]. It then converts the RGB values to XYZ color space using a transformation matrix. The XYZ values are then used to compute the LUV values. The LUV values are scaled to the 8-bit range before being returned.

        The function uses the following formulas for the conversion:
        - u' = 4X / (X + 15Y + 3Z)
        - v' = 9Y / (X + 15Y + 3Z)
        - L = 116 * (Y)^(1/3) - 16 for Y > 0.008856
        - L = 903.3 * Y for Y <= 0.008856
        - u = 13L * (u' - u_n) where u_n is a constant equal to 0.19793943
        - v = 13L * (v' - v_n) where v_n is a constant equal to 0.46831096

        The function finally reshapes the result back to the original image shape and displays the filtered image.
        """

        # Convert RGB to [0, 1] range
        image_rgb = self.image_rgb / 255.0

        # Transformation matrix from RGB to XYZ
        M = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])

        # Reshape the image to a 2D array so we can apply the transformation matrix
        reshaped_rgb = image_rgb.reshape(-1, 3)

        # Convert RGB to XYZ by multiplying with transformation matrix
        xyz = np.dot(reshaped_rgb, M.T)

        # Compute u' and v' from XYZ
        u_prime = 4 * xyz[:, 0] / (xyz[:, 0] + 15 * xyz[:, 1] + 3 * xyz[:, 2])
        v_prime = 9 * xyz[:, 1] / (xyz[:, 0] + 15 * xyz[:, 1] + 3 * xyz[:, 2])

        # Compute L from Y using piecewise function
        L = np.where(xyz[:, 1] > 0.008856, 116 * np.cbrt(xyz[:, 1]) - 16, 903.3 * xyz[:, 1])

        # Compute u and v from L, u', and v'
        u = 13 * L * (u_prime - 0.19793943)
        v = 13 * L * (v_prime - 0.46831096)

        # Scale L, u, v to 8-bit range
        L = 255 / 100 * L
        u = 255 / 354 * (u + 134)
        v = 255 / 262 * (v + 140)

        # Reshape the result back to the original image shape and display the filtered image
        self.display_filtered_img(np.dstack((L, u, v)).reshape(image_rgb.shape).astype(np.uint8), self.output_graphicsView_3, self.output_scene_3)
        
    # --------------------------------Agglomerative Clustering--------------------------------  
    def euclidean_distance(self, x1, x2):
        """
        Computes and returns the Euclidean distance between two points.

        Args:
            x1 (np.ndarray): The first point, represented as a numpy array of coordinates.
            x2 (np.ndarray): The second point, represented as a numpy array of coordinates.

        Returns:
            float: The Euclidean distance between `x1` and `x2`.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def clusters_distance(self, cluster1, cluster2):
        """
        Computes and returns the Euclidean distance between the centroids of two clusters.

        Args:
            cluster1 (List[List[float]]): The first cluster, which is a list of points. Each point is a list of floats representing its coordinates.
            cluster2 (List[List[float]]): The second cluster, which is a list of points. Each point is a list of floats representing its coordinates.

        Returns:
            None
        """
        cluster1_center = np.average(cluster1, axis=0)
        cluster2_center = np.average(cluster2, axis=0)
        return self.euclidean_distance(cluster1_center, cluster2_center)
    
    def initial_clusters(self, points):
        """
        Partitions pixels into self.initial_k groups based on color similarity.

        Args:
            points (List[List[float]]): A list of points where each point is a list of floats representing its coordinates.

        Returns:
            List[List[float]]: A list of groups where each group is a list of points that are similar in color. Only non-empty groups are returned.
        """
        groups = {}
        d = int(256 / self.initial_k)
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            if i % 100000 == 0:
                print('processing pixel:', i)
            go = min(groups.keys(), key=lambda c: self.euclidean_distance(p, c))
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]
    
    def fit_agglomerative(self, points):
        """
        Fits the model to the data in `points`. It starts by assigning each point to a distinct cluster, then repeatedly merges the two most similar clusters until only the desired number of clusters remain.

        Args:
            points (List[List[float]]): A list of points where each point is a list of floats representing its coordinates.

        Returns:
            None
        """
        # initially, assign each point to a distinct cluster
        print('Computing initial clusters ...')
        self.clusters_list = self.initial_clusters(points)
        print('number of initial clusters:', len(self.clusters_list))
        print('merging clusters ...')

        while len(self.clusters_list) > self.clusters_num:
            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: self.clusters_distance(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

            print('number of clusters:', len(self.clusters_list))

        print('assigning cluster num to each point ...')
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        print('Computing cluster centers ...')
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)
            
    def predict_cluster(self, point):
        """
        Finds and returns the cluster number that a given point belongs to.

        Args:
            point (List[float]): The point for which to find the cluster number. The point is a list of floats representing its coordinates.

        Returns:
            int: The cluster number that the point belongs to. The cluster numbers are determined by the `fit` method.
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        """
        Finds and returns the center of the cluster that a given point belongs to.

        Args:
            point (List[float]): The point for which to find the cluster center. The point is a list of floats representing its coordinates.

        Returns:
            List[float]: The center of the cluster that the point belongs to. The center is a list of floats representing its coordinates.
        """
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center
    
    def apply_agglomerative(self, source: np.ndarray, clusters_numbers: int = 2, initial_clusters: int = 25):
        """
        Applies the agglomerative clustering algorithm to an input image and returns the clustered image.

        Args:
            source (np.ndarray): The input image, represented as a numpy array.
            clusters_numbers (int, optional): The number of clusters to form. Defaults to 2.
            initial_clusters (int, optional): The initial number of clusters. Defaults to 25.

        Returns:
            np.ndarray: The clustered image, represented as a numpy array of unsigned 8-bit integers.
        """
        self.clusters_num = clusters_numbers
        self.initial_k = initial_clusters
        src = np.copy(source.reshape((-1, 3)))
        
        self.fit_agglomerative(src)
        
        self.output_image = [[self.predict_center(list(src)) for src in row] for row in source]
        self.output_image = np.array(self.output_image, np.uint8)

        return self.output_image

    def fit(self, data):
        """
        This function applies the Mean Shift algorithm to the data.

        The function first creates a BallTree from the data for efficient nearest neighbor queries. It then enters a loop that continues until all data points have been visited. In each iteration of the loop, the function selects a random data point as the initial mean and then enters another loop where it updates the mean by calculating the mean of all data points within the specified bandwidth of the current mean. If the new mean is close enough to the current mean (as determined by the threshold), the function checks if the new mean is close to any of the existing cluster centers. If it is, it merges the new mean with the closest cluster center. If it isn't, it creates a new cluster center at the location of the new mean. The function then removes all data points within the bandwidth of the new mean from the data and updates the BallTree. If the new mean is not close enough to the current mean, the function updates the current mean to the new mean and repeats the loop.
        """

        # Create a BallTree for efficient nearest neighbor queries
        ball_tree = BallTree(data)

        # Continue until all data points have been visited
        while len(data) > 0:
            # Print the number of remaining data points
            print(f"Remaining data points: {len(data)}")

            # Select a random point in feature space (Initial mean)
            current_mean_index = np.random.randint(0, len(data))
            current_mean = data[current_mean_index]

            # Continue until convergence is achieved
            while True:
                # For uniform window, select points in the range of specified bandwidth and track that points
                tracked_points_indices = ball_tree.query_radius(current_mean.reshape(1, -1), r=self.bandwidth)[0]
                tracked_points = data[tracked_points_indices]

                # Get the new mean, it is the mean of points within bandwidth
                new_mean = np.mean(tracked_points, axis=0)

                # If distance between new and old means < threshold
                if np.linalg.norm(new_mean - current_mean) < self.threshold:
                    # Print a message when convergence is achieved
                    print("Convergence achieved")

                    # Check merge condition
                    for i, c in enumerate(self.clusters):
                        if np.linalg.norm(c - new_mean) < 0.5 * self.bandwidth:
                            # Mean of cluster c = 0.5*distance(c,center)
                            self.clusters[i] = 0.5 * (c + new_mean)
                            break
                    else:
                        # No merge
                        self.clusters.append(new_mean)

                    # Update visited points
                    data = np.delete(data, tracked_points_indices, axis=0)
                    if len(data) == 0:  # Break the loop if data is empty
                        break
                    ball_tree = BallTree(data)  # Update the BallTree
                    break
                else:
                    current_mean = new_mean

    def predict(self, data):
        """
        This function predicts the cluster labels for the given data points.

        The function calculates the distance from each data point to all cluster centers and assigns each data point to the cluster whose center is closest.
        """

        labels = []
        for point in data:
            # Calculate the distance from the point to all cluster centers
            distances = cdist([point], self.clusters)

            # Assign the point to the cluster whose center is closest
            labels.append(np.argmin(distances))

        return labels

    def mean_shift_segmentation(self, image):
        """
        This function applies Mean Shift segmentation to an image.

        The function first reshapes the image to a 2D array of pixels. It then applies the Mean Shift algorithm to the pixels by calling the fit function. It predicts the cluster label for each pixel by calling the predict function. The labels are reshaped to have the same dimensions as the original image. A segmented image is created by assigning each pixel the value of its cluster center. The segmented image is then displayed.
        """

        # Reshape the image to be a 2D array
        pixels = image.reshape(-1, 3)

        # Fit the data
        self.fit(pixels)

        # Predict the labels for each pixel
        labels = self.predict(pixels)

        # Reshape the labels to have the same dimensions as the original image
        labels = np.array(labels).reshape(image.shape[0], image.shape[1])

        # Create a segmented image
        segmented_image = np.zeros_like(image)
        for i in range(len(self.clusters)):
            segmented_image[labels == i] = self.clusters[i]

        cv2.imwrite('mean shift.png', segmented_image)

        # Display the segmented image
        self.display_filtered_img(segmented_image, self.output_graphicsView_2, self.output_scene_2)

        # Compute and print the number of unique colors in the segmented image
        colors = np.unique(segmented_image.reshape(-1, segmented_image.shape[2]), axis=0)
        print(f"# colors in the segmented image is {len(colors)}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
