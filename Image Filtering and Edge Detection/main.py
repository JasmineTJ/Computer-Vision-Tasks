import cv2
import numpy
import numpy as np
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
import sys
from scipy import ndimage
from scipy.ndimage import convolve
import copy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=20, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Define noisy_image as an instance variable
        self.noisy_image = None

        # Load the UI Page
        uic.loadUi(r'gui.ui', self)

        # ui
        self.handle_ui()

        # align text in the label at center
        self.label_hybrid_image.setAlignment(Qt.AlignCenter)

        # Set names for the tabs
        self.tabWidget.setTabText(0, "First Tab")
        self.tabWidget.setTabText(1, "Histograms")
        self.tabWidget.setTabText(2, "Hybrid Image")

        # Set the first tab as the initially selected tab
        self.tabWidget.setCurrentIndex(0)

        # Connect the mouseDoubleClickEvent of original_image to custom function
        self.original_image.mouseDoubleClickEvent = lambda event: self.browse_image(event , 0)
        self.image_1.mouseDoubleClickEvent = lambda event: self.browse_image(event, 1)
        self.image_2.mouseDoubleClickEvent = lambda event: self.browse_image(event, 2)
        # img= cv2.imread('task1/Jerry_Mouse.png')
        # img = self.rgbtogray(img)
        # g= self.prewit(img)
        # cv2.imwrite('test2.jpg', np.real(g))
        self.edge_combo.currentTextChanged.connect(self.edge_detection)

        self.freq_filter_combo.currentTextChanged.connect(self.freq_domain_filter)

        # initialize variables for matplotlib plotting
        self.canvas1, self.layout1 = self.create_canvas_layout(self.histogram_graph)
        # Set the layout for the respective widgets
        self.histogram_graph.setLayout(self.layout1)

        self.canvas2, self.layout2 = self.create_canvas_layout(self.distribution_curve_graph)
        self.distribution_curve_graph.setLayout(self.layout2)

        self.canvas3,self.layout3 = self.create_canvas_layout(self.red_histo_graph)
        self.red_histo_graph.setLayout(self.layout3)

        self.canvas4, self.layout4 = self.create_canvas_layout(self.green_histo_graph)
        self.green_histo_graph.setLayout(self.layout4)

        self.canvas5, self.layout5 = self.create_canvas_layout(self.blue_histo_graph)
        self.blue_histo_graph.setLayout(self.layout5)

        self.canvas6, self.layout6 = self.create_canvas_layout(self.red_dist_graph)
        self.red_dist_graph.setLayout(self.layout6)

        self.canvas7, self.layout7 = self.create_canvas_layout(self.green_dist_graph)
        self.green_dist_graph.setLayout(self.layout7)

        self.canvas8, self.layout8 = self.create_canvas_layout(self.blue_dist_graph)
        self.blue_dist_graph.setLayout(self.layout8)

        self.canvas9, self.layout9 = self.create_canvas_layout(self.cumulative_curve)
        self.cumulative_curve.setLayout(self.layout9)

        # Connect currentIndexChanged signal of noise_combo to noise_combo_currentIndexChanged slot
        self.noise_combo.currentIndexChanged.connect(self.noise_combo_currentIndexChanged)

        # Connect currentIndexChanged signal of comboBox to comboBox_currentIndexChanged slot
        self.filter_combo.currentIndexChanged.connect(self.filter_combo_currentIndexChanged)

        self.pixmap_size = None
        self.images_to_mix = []

    def create_canvas_layout(self, graph_widget):
        canvas = MplCanvas(graph_widget, width=20, height=5, dpi=100)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(canvas)
        return canvas,layout

    def handle_ui(self):
        # Create a QGraphicsScene
        self.original_scene = QtWidgets.QGraphicsScene()
        self.original_image.setScene(self.original_scene)
        
        self.filtered_scene = QtWidgets.QGraphicsScene()
        self.filtered_image.setScene(self.filtered_scene)
        
        self.global_thresholding_scene = QtWidgets.QGraphicsScene()
        self.global_thresholding.setScene(self.global_thresholding_scene)
        
        self.local_thresholding_scene = QtWidgets.QGraphicsScene()
        self.local_thresholding.setScene(self.local_thresholding_scene)

        self.filtered_scene = QtWidgets.QGraphicsScene()
        self.filtered_image.setScene(self.filtered_scene)
        
        self.hybrid_image_scene = QtWidgets.QGraphicsScene()
        self.hybrid_image.setScene(self.hybrid_image_scene)

        self.image_1_scene = QtWidgets.QGraphicsScene()
        self.image_1.setScene(self.image_1_scene)

        self.image_2_scene = QtWidgets.QGraphicsScene()
        self.image_2.setScene(self.image_2_scene)

        self.equalization_scene = QtWidgets.QGraphicsScene()
        self.equalization.setScene(self.equalization_scene)

        self.normalization_scene = QtWidgets.QGraphicsScene()
        self.normalization.setScene(self.normalization_scene)

        self.hybrid_image_scene = QtWidgets.QGraphicsScene()
        self.hybrid_image.setScene(self.hybrid_image_scene)

    def filter_combo_currentIndexChanged(self, index):
        """
        Handle the change in selection of filter type.

        Parameters:
            index (int): Index of the selected filter type in the combo box.

        Description:
            This method is triggered when the user selects a different filter type from the combo box. It determines
            the selected filter type based on the index and calls the corresponding method to apply the filter.

        Example:
            # Assuming index is the index of the selected filter type.
            filter_combo_currentIndexChanged(index)
        """

        index = self.filter_combo.currentIndex()
        if index == 0:
            self.show_noisy_image()
        elif index == 1:
            self.apply_Average_filter()
        elif index == 2:
            self.apply_Gaussian_filter()
        elif index == 3:
            self.apply_Median_filter()

    def noise_combo_currentIndexChanged(self, index):
        """
        Handle the change in selection of noise type.

        Parameters:
            index (int): Index of the selected noise type in the combo box.

        Description:
            This method is triggered when the user selects a different noise type from the combo box. It determines
            the selected noise type based on the index and calls the corresponding method to apply the noise.

        Example:
            # Assuming index is the index of the selected noise type.
            noise_combo_currentIndexChanged(index)
        """

        self.filter_combo.setCurrentIndex(0)

        index = self.noise_combo.currentIndex()

        if index == 0:
            self.show_same_image()

        if index == 1:
            self.apply_uniform_noise()

        if index == 2:
            self.apply_Gaussian_noise()

        if index == 3:
            self.apply_salt_and_pepper_noise()

    def show_same_image(self):
        """
        Show the original image on the filtered image view.

        Description:
            This method retrieves the original image from self.original_scene, converts it to a NumPy array,
            and displays it on self.filtered_image.

        Example:
            self.show_same_image()
        """
        # Take the image from self.original_image and show it on self.filtered_image
        pixmap_item = self.original_scene.items()[0]
        pixmap = pixmap_item.pixmap()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format_RGB888)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)

        # Convert the image array to QPixmap
        qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Create a QGraphicsPixmapItem with the original image
        pixmap_item = QGraphicsPixmapItem(pixmap)

        # Clear the existing scene in self.filtered_scene
        self.filtered_scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.filtered_scene.clear()

        # Add the original pixmap item to the scene of self.filtered_image
        self.filtered_scene.addItem(pixmap_item)

        # Resize the QGraphicsView to fit the scene without scroll bars
        self.filtered_image.fitInView(self.filtered_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def show_noisy_image(self):
        """
        Show the noisy image on the filtered image view.

        Description:
            This method retrieves the noisy image from self.noisy_image, converts it to a QPixmap, and displays
            it on self.filtered_image.

        Example:
            self.show_noisy_image()
        """
        if self.filtered_scene.items():
            # Get the first item from self.filtered_scene
            pixmap_item = self.filtered_scene.items()[0]
            pixmap = pixmap_item.pixmap()
            img = pixmap.toImage()
            img = img.convertToFormat(QImage.Format_RGB888)
            width, height = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = self.noisy_image

            # Convert the image array to QPixmap
            qimg = QImage(arr.data, arr.shape[1], arr.shape[0], arr.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # Create a QGraphicsPixmapItem with the original image
            pixmap_item = QGraphicsPixmapItem(pixmap)

            # Clear the existing scene in self.filtered_scene
            self.filtered_scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            self.filtered_scene.clear()

            # Add the original pixmap item to the scene of self.filtered_image
            self.filtered_scene.addItem(pixmap_item)

            # Resize the QGraphicsView to fit the scene without scroll bars
            self.filtered_image.fitInView(self.filtered_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            # Handle the case where the scene is empty
            print("Filtered scene is empty. Cannot show noisy image.")

    def apply_Average_filter(self):
        """
        Apply average filter to the currently displayed noisy image.

        Description:
            This method retrieves the currently displayed noisy image from self.noisy_image. It converts the image to a
            NumPy array and applies average filter using the convolve1d method. The filtered image is then displayed
            using draw_filtered_image method.

        Example:
            # Assuming self.noisy_image is an instance variable containing the noisy image
            # and self.convolve1d is a method for applying convolution.
            # Assuming self.draw_filtered_image is a method to display the filtered image.
            self.apply_Average_filter()
        """

        # Get the currently displayed image from self.filtered_scene
        pixmap_item = self.filtered_scene.items()[0]
        pixmap = pixmap_item.pixmap()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format_RGB888)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = self.noisy_image

        # Define the average filter kernel
        kernel = np.ones((3, 3)) / 9  # 3x3 average filter

        # Apply convolution to each channel of the image separately
        filtered_image = np.zeros_like(arr)
        for c in range(3):  # Iterate over each channel (RGB)
            filtered_image[:, :, c] = self.convolve1d(arr[:, :, c], kernel)

        self.draw_filtered_image(self.noisy_image)

    def convolve1d(self, img, kernel):
        """
        Apply 1D convolution to the image.

        Parameters:
            img (numpy.ndarray): Input image.
            kernel (numpy.ndarray): Convolution kernel.

        Returns:
            numpy.ndarray: Filtered image.

        Description:
            This method applies 1D convolution to the input image using the specified kernel.

        Example:
            # Assuming img is a NumPy array representing the image and kernel is the convolution kernel.
            filtered_img = convolve1d(img, kernel)
        """
        # Get the dimensions of the image and the kernel
        img_height, img_width = img.shape
        kernel_width = kernel.shape[0]

        # Calculate the padding required
        pad_width = kernel_width // 2

        # Create an array to hold the result of the convolution
        result = np.zeros_like(img)

        # Apply padding to the image
        padded_img = np.pad(img, ((0, 0), (pad_width, pad_width)), mode='constant')

        # Perform 1D convolution
        for y in range(img_height):
            for x in range(img_width):
                # Extract the region of interest (ROI) from the padded image
                roi = padded_img[y, x:x + kernel_width]

                # Perform element-wise multiplication between the ROI and the kernel
                convolution = np.sum(roi * kernel)

                # Store the result in the output array
                result[y, x] = convolution

        return result


    def apply_Gaussian_filter(self):

        """
        Apply Gaussian filter to the currently displayed noisy image.

        Description:
            This method retrieves the currently displayed noisy image from self.noisy_image. It converts the image to a
            NumPy array and applies Gaussian filter using the gaussian_kernel and convolve2d methods. The filtered image
            is then displayed using draw_filtered_image method.

        Example:
            # Assuming self.noisy_image is an instance variable containing the noisy image
            # and self.gaussian_kernel and self.convolve2d are methods for generating Gaussian kernel and applying
            # convolution, respectively.
            # Assuming self.draw_filtered_image is a method to display the filtered image.
            self.apply_Gaussian_filter()
        """
        # Get the currently displayed image from self.filtered_scene
        pixmap_item = self.filtered_scene.items()[0]
        pixmap = pixmap_item.pixmap()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format_RGB888)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = self.noisy_image

        # Define the Gaussian filter kernel
        kernel = self.gaussian_kernel(size=5, sigma=1.5)  # 5x5 Gaussian filter

        # Apply convolution to each channel of the image separately
        filtered_image = np.zeros_like(arr)
        for c in range(3):  # Iterate over each channel (RGB)
            filtered_image[:, :, c] = self.convolve2D(arr[:, :, c], kernel)
        self.draw_filtered_image(self.noisy_image)

    def gaussian_kernel(self, size, sigma):
        """
        Generates a Gaussian kernel.
        :param size: Size of the kernel (odd number).
        :param sigma: Standard deviation of the Gaussian distribution.
        :return: Gaussian kernel.
        """
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
        kernel /= np.sum(kernel)
        return kernel

    def apply_Median_filter(self):
        """
        Apply median filter to the currently displayed noisy image.

        Description:
            This method retrieves the currently displayed noisy image from self.noisy_image. It converts the image to a
            NumPy array and applies median filter to each channel of the image separately using the median_filter method.
            The filtered image is then displayed using draw_filtered_image method.

        Example:
            # Assuming self.noisy_image is an instance variable containing the noisy image
            # and self.median_filter is a method for applying median filter.
            # Assuming self.draw_filtered_image is a method to display the filtered image.
            self.apply_Median_filter()
        """
        # Get the currently displayed image from self.filtered_scene
        pixmap_item = self.filtered_scene.items()[0]
        pixmap = pixmap_item.pixmap()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format_RGB888)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = self.noisy_image
        # Apply median filter to each channel of the image separately
        filtered_image = np.zeros_like(arr)
        for c in range(3):  # Iterate over each channel (RGB)
            filtered_image[:, :, c] = self.median_filter(arr[:, :, c], size=3)

        self.draw_filtered_image(self.noisy_image)

    def median_filter(self, img, size):
        """
        Applies median filter to the image.
        :param img: Input image.
        :param size: Size of the median filter kernel.
        :return: Filtered image.
        """
        pad = size // 2
        img_pad = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')
        filtered_img = np.zeros_like(img)
        for i in range(filtered_img.shape[0]):
            for j in range(filtered_img.shape[1]):
                filtered_img[i, j] = np.median(img_pad[i:i + size, j:j + size])
        return filtered_img

    def apply_uniform_noise(self):
        """
        Apply uniform noise to the currently displayed image.

        Description:
            This method retrieves the currently displayed image from self.original_image. It converts the image to a
            NumPy array and applies uniform noise using the uniform_noise method. The noisy image is then displayed
            using draw_filtered_image method.

        Example:
            # Assuming self.original_image is an instance variable containing the original image
            # and self.uniform_noise is a method for applying uniform noise.
            # Assuming self.draw_filtered_image is a method to display the filtered image.
            self.apply_uniform_noise()
        """
        # Get the currently displayed image from self.original_image
        pixmap_item = self.original_scene.items()[0]
        pixmap = pixmap_item.pixmap()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format_RGB888)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)

        # Apply uniform noise to the image
        self.noisy_image = self.uniform_noise(arr)
        self.draw_filtered_image(self.noisy_image)

    def uniform_noise(self, img):
        """
        Add uniform noise to the image.

        Parameters:
            img (numpy.ndarray): A NumPy array representing the image.

        Returns:
            numpy.ndarray: A NumPy array representing the noisy image.

        Description:
            This method adds uniform noise to the input image. It generates uniform noise for each channel separately
            and adds it to the corresponding pixel values of the image.

        Example:
            # Assuming img is a NumPy array representing the image.
            noisy_img = uniform_noise(img)
        """
        row, col, _ = img.shape
        uni_noise = np.zeros((row, col, 3), dtype=np.uint8)  # Modify shape to include the third dimension
        cv2.randu(uni_noise, 0, 255)
        uni_noise = (uni_noise * 0.5).astype(np.uint8)
        noisy_img = img + uni_noise
        return noisy_img

    def apply_Gaussian_noise(self):
        """
        Apply Gaussian noise to the currently displayed image.

        Description:
            This method retrieves the currently displayed image from self.original_image. It converts the image to a
            NumPy array and applies Gaussian noise using the gaussian_noise method. The noisy image is then displayed
            using draw_filtered_image method.

        Example:
            # Assuming self.original_image is an instance variable containing the original image
            # and self.gaussian_noise is a method for applying Gaussian noise.
            # Assuming self.draw_filtered_image is a method to display the filtered image.
            self.apply_Gaussian_noise()
        """
        # Get the currently displayed image from self.original_image
        pixmap_item = self.original_scene.items()[0]
        pixmap = pixmap_item.pixmap()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format_RGB888)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)

        # Apply Gaussian noise to the image
        self.noisy_image = self.gaussian_noise(arr)
        self.draw_filtered_image(self.noisy_image)

    def gaussian_noise(self, img):
        """
        Add Gaussian noise to the image.

        Parameters:
            img (numpy.ndarray): A NumPy array representing the image.

        Returns:
            numpy.ndarray: A NumPy array representing the noisy image.

        Description:
            This method adds Gaussian noise to the input image. It generates Gaussian noise for each channel separately
            and adds it to the corresponding pixel values of the image. The pixel values are then clipped to ensure they
            stay within the valid range (0-255) and converted to uint8 data type.

        Example:
            # Assuming img is a NumPy array representing the image.
            noisy_img = gaussian_noise(img)
        """
        row, col, channels = img.shape
        mean = 0
        var = 3000  # Adjust the variance to a smaller value
        sigma = var ** 0.5
        # Generate Gaussian noise for each channel separately
        gaussian_noise = np.random.normal(mean, sigma, size=(row, col, channels))
        noisy_img = img + gaussian_noise
        # Clip pixel values to stay within the valid range
        noisy_img = np.clip(noisy_img, 0, 255)
        # Ensure the dtype is uint8 to represent pixel values
        noisy_img = noisy_img.astype(np.uint8)
        return noisy_img

    def apply_salt_and_pepper_noise(self):
        """
        Apply salt and pepper noise to a given image.

        Parameters:
            img (numpy.ndarray): A NumPy array representing the image to which noise is applied.

        Returns:
            numpy.ndarray: A NumPy array representing the noisy image.

        Description:
            This function copies the original image to a new array. It then iterates through each pixel in the image
            and assigns either black (pepper noise) or white (salt noise) to it based on random probabilities. The
            result is a noisy image array with salt and pepper noise.

        Example:
            >>> import numpy as np
            >>> original_image = np.array([[0, 255, 0],
                                        [255, 0, 255],
                                        [0, 255, 0]], dtype=np.uint8)
            >>> noisy_image = apply_salt_and_pepper_noise(original_image)
            >>> print(noisy_image)
            [[255   0 255]
            [  0 255   0]
            [255   0 255]]
        """
        # Get the currently displayed image from self.original_image
        pixmap_item = self.original_scene.items()[0]
        pixmap = pixmap_item.pixmap()
        img = pixmap.toImage()
        img = img.convertToFormat(QImage.Format_RGB888)
        width, height = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)

        # Apply salt and pepper noise to the image
        self.noisy_image = self.salt_and_pepper(arr)
        self.draw_filtered_image(self.noisy_image)

    def salt_and_pepper(self, img):
        """
        Apply salt and pepper noise to the image.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with salt and pepper noise.

        Description:
            This method applies salt and pepper noise to the input image. It iterates through each pixel of the image
            and assigns either black (pepper noise) or white (salt noise) to it based on predefined probabilities.

        Example:
            # Assuming img is a NumPy array representing the image.
            noisy_img = salt_and_pepper(img)
        """
        row, col, channels = img.shape
        noisy_img = img.copy()  # Create a copy of the original image
        salt = 0.95
        pepper = 0.1
        for i in range(row):
            for j in range(col):
                rdn = np.random.random()
                if rdn < pepper:
                    noisy_img[i, j] = [0, 0, 0]  # Set pixel to black for pepper noise
                elif rdn > salt:
                    noisy_img[i, j] = [255, 255, 255]  # Set pixel to white for salt noise
        return noisy_img

    def draw_filtered_image(self, filtered_image):
        """
        Draw the filtered image on the QGraphicsView.

        Parameters:
            filtered_image (numpy.ndarray): Filtered image array.

        Description:
            This method converts the filtered image array to QPixmap and adds it to the QGraphicsScene.
            It then resizes the QGraphicsView to fit the scene.

        Example:
            # Assuming filtered_image is a NumPy array representing the filtered image.
            draw_filtered_image(filtered_image)
        """
        # Convert the filtered image array to QPixmap
        qimg = QImage(filtered_image.data, filtered_image.shape[1], filtered_image.shape[0], filtered_image.strides[0],
                      QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Create a QGraphicsPixmapItem with the filtered image
        pixmap_item = QGraphicsPixmapItem(pixmap)

        # Clear the existing scene in self.filtered_scene
        self.filtered_scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.filtered_scene.clear()

        # Add the filtered pixmap item to the scene of self.filtered_image
        self.filtered_scene.addItem(pixmap_item)

        # Resize the QGraphicsView to fit the scene without scroll bars
        self.filtered_image.fitInView(self.filtered_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def browse_image(self, event, index):
        """
        Browse and load an image file.

        Parameters:
            event: The event object triggering the method.
            index (int): Index corresponding to the image type.

        Description:
            This method opens a file dialog for the user to select an image file. It loads the selected image file,
            converts it to QPixmap, and adds it to the appropriate QGraphicsScene. It also resizes the QGraphicsView
            to fit the loaded image.

        Example:
            # Assuming event is the event object triggering the method and index is the index corresponding to the image type.
            browse_image(event, index)
        """
        # Clear the existing scene in self.filtered_scene
        self.filtered_scene.clear()
        self.noise_combo.setCurrentIndex(0)
        self.filter_combo.setCurrentIndex(0)

        # Open a file dialog to select an image file
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]

            # Load the selected image file using OpenCV
            image_unchanged = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Convert the image to RGB for QPixmap
            image_rgb = cv2.cvtColor(image_unchanged, cv2.COLOR_BGR2RGB if len(image_unchanged.shape) == 3 else cv2.COLOR_GRAY2RGB)

            # Create QImage from the RGB image
            height, width = image_rgb.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Create QPixmap from QImage
            pixmap = QPixmap.fromImage(q_image)

            # Create a QGraphicsPixmapItem with the loaded image
            pixmap_item = QGraphicsPixmapItem(pixmap)

            if index == 0:
                # Clear the existing scene
                self.original_scene.clear()

                # Set the scene's dimensions to match the dimensions of the loaded image
                self.original_scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

                # Add the pixmap item to the scene
                self.original_scene.addItem(pixmap_item)

                # Resize the QGraphicsView to fit the scene
                self.original_image.fitInView(self.original_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
                
                # Store the size of the original pixmap
                self.pixmap_size = pixmap.size()

                # get current image
                image = self.get_current_orignal_img()
                
                # gray image
                rgbtogray_img = self.rgbtogray(image)
                gray_img = rgbtogray_img.astype(np.uint8) 
                threshold_value = self.otsu_threshold_value(gray_img)
                # global thresholding
                global_img = self.global_thresholding_fn(gray_img, threshold_value)
                cv2.imwrite('global_img.jpg', np.real(global_img))
                global_img = cv2.imread('global_img.jpg', cv2.IMREAD_GRAYSCALE)
                self.display_filtered_img(global_img, self.global_thresholding, self.global_thresholding_scene)
                
                # local thresholding
                estimated_character_size = 20  
                window_size = self.estimate_local_window_size(estimated_character_size)
                print(f"Estimated local window size: {window_size} pixels")
                local_img = self.local_thresholding_fn(rgbtogray_img, window_size)
                cv2.imwrite('local_img.jpg', np.real(local_img))
                local_img = cv2.imread('local_img.jpg', cv2.IMREAD_GRAYSCALE)
                self.display_filtered_img(local_img, self.local_thresholding, self.local_thresholding_scene)
                
                self.clear_canvas()
                self.draw_histogram(image)
                self.cumm_dist(image)
                self.rgb_histogram(image)
                self.equalize_image(image_unchanged)
                self.normalize_image(image)


            if index == 1:
                # Clear the existing scene
                self.image_1_scene.clear()

                # Set the scene's dimensions to match the dimensions of the loaded image
                self.image_1_scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

                # Add the pixmap item to the scene
                self.image_1_scene.addItem(pixmap_item)

                # Resize the QGraphicsView to fit the scene
                self.image_1.fitInView(self.image_1_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

                # Store the size of the img1 pixmap
                self.pixmap_size = pixmap.size()

                # Ensure self.images_to_mix has enough elements
                while len(self.images_to_mix) < 1:
                    self.images_to_mix.append(None)

                # Replace the first element in self.images_to_mix with image_unchanged
                self.images_to_mix[0] = image_unchanged

                image = cv2.imread(file_path)
                # self.clear_canvas()
                # self.draw_histogram(image)
                # self.cumm_dist(image)
                # self.rgb_histogram(image)

                if len(self.images_to_mix) == 2:
                    self.create_hybrid_image()

            if index == 2:
                # Clear the existing scene
                self.image_2_scene.clear()

                # Set the scene's dimensions to match the dimensions of the loaded image
                self.image_2_scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

                # Add the pixmap item to the scene
                self.image_2_scene.addItem(pixmap_item)

                # Resize the QGraphicsView to fit the scene
                self.image_2.fitInView(self.image_2_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

                # Ensure self.images_to_mix has enough elements
                while len(self.images_to_mix) < 2:
                    self.images_to_mix.append(None)

                # Replace the second element in self.images_to_mix with image_unchanged
                self.images_to_mix[1] = image_unchanged

                image = cv2.imread(file_path)
                # self.clear_canvas()
                # self.draw_histogram(image)
                # self.cumm_dist(image)
                # self.rgb_histogram(image)

                if len(self.images_to_mix) == 2:
                    self.create_hybrid_image()
                
    # --------------------------------Rgb to Gray--------------------------------  
    
    def rgbtogray(self, image):
        """
        Converts an RGB image to grayscale.

        Args:
            image (numpy.ndarray): Input RGB image.

        Returns:
            numpy.ndarray: Grayscale image.
        """
        r,g,b=image[:,:,0],image[:,:,1],image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
        
    # --------------------------------Edge Detection--------------------------------  
     
    #                     ------------Sobel Filter------------   
    
    def sobel(self, img):
        """
        Applies the Sobel edge detection filter to an input image.

        Args:
            img (numpy.ndarray): Input grayscale image.

        Returns:
            numpy.ndarray: Image with Sobel edges highlighted.
        """
        # creates a 3x3 kernel (filter) for the x-direction Sobel filter
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        # creates a 3x3 kernel for the y-direction Sobel filter
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        # the gradient in the x-direction
        Ix = convolve(img, Kx)
        # the gradient in the y-direction
        Iy = convolve(img, Ky)
        
        # computes the magnitude of the gradient using the Euclidean distance between Ix and Iy
        G = np.hypot(Ix, Iy)
        # normalizes the gradient magnitude values to the range [0, 255]
        G = G / G.max() * 255
        
        return G
    
    #                     ------------Robert Filter------------ 
    
    def robert(self, img):
        """
        Applies the Roberts edge detection filter to an input image.

        Args:
            img (numpy.ndarray): Input grayscale image.

        Returns:
            numpy.ndarray: Image with Roberts edges highlighted.
        """
        roberts_cross_v = np.array( [[ 0, 0, 0 ],
                                [ 0, 1, 0 ],
                                [ 0, 0,-1 ]] )

        roberts_cross_h = np.array( [[ 0, 0, 0 ],
                                [ 0, 0, 1 ],
                                [ 0,-1, 0 ]] )
        vertical = ndimage.convolve(img, roberts_cross_v)
        horizontal = ndimage.convolve(img, roberts_cross_h)
        edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
        return edged_img
    
    #                     ------------Prewit Filter------------ 
    
    def prewit(self, img):
        """
        Applies the Prewitt edge detection filter to an input image.

        Args:
            img (numpy.ndarray): Input grayscale image.

        Returns:
            numpy.ndarray: Image with Prewitt edges highlighted.
        """
        # define horizontal and Vertical prewit kernels
        Hx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
        Hy = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
        # normalizing the vectors
        pre_x = self.convolve2D(img, Hx) / 6.0
        pre_y = self.convolve2D(img, Hy) / 6.0
        # calculate the gradient magnitude of vectors
        pre_out = np.sqrt(np.power(pre_x, 2) + np.power(pre_y, 2))
        # mapping values from 0 to 255
        pre_out = (pre_out / np.max(pre_out)) * 255
        return pre_out
    
    #                     ------------Canny Filter------------ 
    # step 1: image in gray level
    # step 2: apply gaussian filter for noise reduction(gaussian smoothing)
    def get_gaussian_kernel(self, kernal_size, sigma=1):
        """
        Generates a Gaussian kernel for noise reduction (Gaussian smoothing).

        Args:
            kernal_size (int): Size of the kernel (odd number).
            sigma (float, optional): Standard deviation for Gaussian distribution. Defaults to 1.

        Returns:
            numpy.ndarray: Gaussian kernel normalized to sum to 1.
        """
        gaussian_kernal = np.zeros((kernal_size, kernal_size), np.float32)
        size = kernal_size//2

        for x in range(-size, size+1):
            for y in range(-size, size+1):
                a = 1/(2*np.pi*(sigma**2))
                b = np.exp(-(x**2 + y**2)/(2* sigma**2))
                gaussian_kernal[x+size, y+size] = a*b
        return gaussian_kernal/gaussian_kernal.sum()
    
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
                image_copy[i].insert(0, input_image[i][-1-j])
                image_copy[i].append(input_image[i][j])
        for i in range(kernal_half):
            image_copy.append(image_copy[2*i])
            image_copy.insert(0, image_copy[-2-2*i].copy())

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
        if (gradient_estimation_filter_type=="sobel"):
            Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
            My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        elif (gradient_estimation_filter_type=="prewitt"):
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
    
    # step 4 : non-maximal suppression 
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

        
        for i in range(1, rows_count-1):
            for j in range(1, columns_count-1):
                next = 255
                previous = 255
                if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
                    next = image[i, j+1]
                    previous = image[i, j-1]
                elif (22.5 <= theta[i,j] < 67.5):
                    next = image[i+1, j-1]
                    previous = image[i-1, j+1]
                elif (67.5 <= theta[i,j] < 112.5):
                    next = image[i+1, j]
                    previous = image[i-1, j]
                elif (112.5 <= theta[i,j] < 157.5):
                    next = image[i-1, j-1]
                    previous = image[i+1, j+1]

                if (image[i,j] >= next) and (image[i,j] >= previous):
                    output_image[i,j] = image[i,j]
                else:
                    output_image[i,j] = 0
        
        return output_image
    
    # step 5 : hysteresis thresholding 
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
        for i in range (len(image)):
            for j in range (len(image[0])):
                if (image[i,j]>=high_threshold):
                    strong_i.append(i)
                    strong_j.append(j)
                if ((image[i,j] <= high_threshold) & (image[i,j] >= low_threshold)):
                    weak_i.append(i)
                    weak_j.append(j)
        strong_i = np.array(strong_i)
        strong_j = np.array(strong_j)
        weak_i = np.array(weak_i)
        weak_j = np.array(weak_j)
        
        output_image[strong_i, strong_j] = strong
        output_image[weak_i, weak_j] = weak

        return (output_image, weak, strong)
    
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
        for i in range(1, rows_count-1):
            for j in range(1, columns_count-1):
                if (image[i,j] == weak):
                    if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                        or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                        or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
        return image
    
    def canny(self, img):
        """
        Applies the Canny edge detection algorithm to an input image.

        Args:
            img (numpy.ndarray): Input grayscale image.

        Returns:
            numpy.ndarray: Image with Canny edges highlighted.
        """
        kernal_size = 3
        low_threshold_ratio = 0.05
        high_threshold_ratio = 0.09
        gradient_estimation_filter_type = "sobel"

        # step 2 : apply gaussian kernal to filter noise
        kernal = self.get_gaussian_kernel(kernal_size)
        image_without_noise = self.apply_filtering(img.tolist(), kernal)

        # step 3 : gradient estimation
        assert (gradient_estimation_filter_type in ["sobel", "prewitt", "robert"]), "gradient estimation filter type should be [\"prewitt\", \"sobel\", \"robert\"]"
        G, theta = self.gradient_estimate(image_without_noise, gradient_estimation_filter_type)

        # step 4 : non maximal suppression
        image_with_thin_edges = self.non_maximal_suppression(G, theta)

        # step 5 : double threshold and hysteresis thresholding
        final_image, weak, strong = self.double_threshold(image_with_thin_edges, low_threshold_ratio, high_threshold_ratio)

        # edge tracking with hysteresis
        img = self.hysteresis_edge_track(final_image, weak, strong=255)
        return img
    
    # --------------------------------Display Filtered Image--------------------------------  
        
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


    def get_current_orignal_img(self):
        """
        Retrieves the currently displayed image from the QGraphicsView.

        Returns:
            numpy.ndarray or None: The image data (RGB format) if an image is loaded, or None if no image is loaded.
        """
        # Check if there is an image loaded
        if self.original_scene.items():
            # Get the currently displayed image from self.original_image
            pixmap_item = self.original_scene.items()[0]
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
    
    def edge_detection(self):
        """
        Performs edge detection on the currently displayed image based on the selected edge detection method.

        Returns:
            None
        """
        # get current original image
        curr_image = self.get_current_orignal_img()
        if self.original_scene.items():
            # get current text
            self.current_text = self.edge_combo.currentText()
            print("self.current_text:", self.current_text)
            rgbtogray_img = self.rgbtogray(curr_image) 
            if self.current_text == "None":
                self.filtered_scene.clear()
                return None
            elif self.current_text == "Sobel":
                filtered_image = self.sobel(rgbtogray_img)
                cv2.imwrite('sobel.jpg', np.real(filtered_image))
                filtered_image = cv2.imread('sobel.jpg', cv2.IMREAD_GRAYSCALE)
            elif self.current_text == "Roberts":
                filtered_image = self.robert(rgbtogray_img)
                cv2.imwrite('robert.jpg', np.real(filtered_image))
                filtered_image = cv2.imread('robert.jpg', cv2.IMREAD_GRAYSCALE)
            elif self.current_text == "Prewitt":
                filtered_image = self.prewit(rgbtogray_img)
                cv2.imwrite('prewit.jpg', np.real(filtered_image))
                filtered_image = cv2.imread('prewit.jpg', cv2.IMREAD_GRAYSCALE)
            else:
                filtered_image = self.canny(rgbtogray_img)
                cv2.imwrite('canny.jpg', np.real(filtered_image))
                filtered_image = cv2.imread('canny.jpg', cv2.IMREAD_GRAYSCALE)
            
            self.display_filtered_img(filtered_image, self.filtered_image, self.filtered_scene)
                
    
    # --------------------------------Convolution--------------------------------  
    
    def convolve2D(self, X, F):
        """
        Performs 2D convolution between an input image and a filter (kernel).

        Args:
            X (numpy.ndarray): Input image.
            F (numpy.ndarray): Filter (kernel).

        Returns:
            numpy.ndarray: Convolved output image.
        """
        # height and width of the image
        X_height = X.shape[0]
        X_width = X.shape[1]
        
        # height and width of the filter
        F_height = F.shape[0]
        F_width = F.shape[1]
        
        H = (F_height - 1) // 2
        W = (F_width - 1) // 2
        
        # output numpy matrix with height and width
        out = np.zeros((X_height, X_width))
        # iterate over all the pixel of image X
        for i in np.arange(H, X_height-H):
            for j in np.arange(W, X_width-W):
                sum = 0
                # iterate over the filter
                for k in np.arange(-H, H+1):
                    for l in np.arange(-W, W+1):
                        # get the corresponding value from image and filter
                        a = X[i+k, j+l]
                        w = F[H+k, W+l]
                        sum += (w * a)
                out[i,j] = sum
                
        # return convolution  
        return out
    
    # --------------------------------Thresholding--------------------------------
    
    def global_thresholding_fn(self, img, global_threshold):
        """
        Performs global thresholding on a grayscale image.

        Args:
            img (numpy.ndarray): The input grayscale image.
            global_threshold (int): The threshold value for binarization.

        Returns:
            numpy.ndarray: The resulting binary image after global thresholding.
        """
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        for row in range(height):  
            for column in range(width):
                if img[row, column] > global_threshold: 
                    img[row, column] = 255
                else:
                    img[row, column] = 0
        return img
    
    def otsu_threshold_value(self, gray_img):
        """
        Calculates the Otsu threshold value for global thresholding.

        Args:
            gray_img (numpy.ndarray): Grayscale image.

        Returns:
            float: Otsu threshold value.
        """
        # Apply Otsu's method to the grayscale image
        _, img_thresholded = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Print the threshold value
        print("Otsu's Threshold Value:", _)
        
        return _
    
    def local_thresholding_fn(self, img, local_window):
        """
        Performs local thresholding on a grayscale image using a specified local window size.

        Args:
            img (numpy.ndarray): The input grayscale image.
            local_window (int): The local window size for thresholding.

        Returns:
            numpy.ndarray: The resulting binary image after local thresholding.
        """
        local_window = local_window + 1
        image = np.zeros_like(img)
        max_row, max_col = img.shape
        for i in range(max_row):
            y_min = max(0, i - local_window)
            y_max = min(max_row, i + local_window + 1)
            for j in range(max_col):
                x_min = max(0, j - local_window)
                x_max = min(max_col, j + local_window + 1)
                window = img[y_min:y_max, x_min:x_max]
                local_thresh = np.median(window)
                if img[i, j] >= local_thresh:
                    image[i, j] = 255
        return image

    def estimate_local_window_size(self, object_size_estimate, scale_factor=2):
        """
        Estimates an initial local window size for adaptive thresholding.

        Args:
            object_size_estimate (int): The assumed size of objects (e.g., characters).
            scale_factor (float, optional): A factor to adjust the initial window size (default is 2).

        Returns:
            int: The initial estimated local window size.
        """
        # Compute an initial window size based on the estimated object size
        initial_window_size = int(object_size_estimate * scale_factor)

        return initial_window_size
    
    # --------------------------------Histogram--------------------------------
    
    def draw_histogram(self, image):
        """
        Draws a histogram of pixel intensity values for the given image.

        Parameters:
            image (numpy.ndarray): The input image.

        Returns:
            None
        """
        # convert image to numpy array
        image_arr=numpy.array(image)
        # convert image from 2d array to 1d array
        flat_image=image_arr.flatten()
        # plot histogram
        self.canvas1.axes.hist(flat_image, bins=50,label="histogrm")
        # add x and y labels
        self.handle_canvas(self.canvas1,'Intensity Value','pixels')

    def cumm_dist(self,image):
        """
        Plots the cumulative distribution of pixel intensity values for the given image.

        Parameters:
            image : The input image.

        Returns:
            None
        """
        # convert image from 2d array to 1d array then plot cumulative distribution
        self.canvas2.axes.hist(image.flatten(), bins=256, cumulative=True, label="dist_curve")
        self.handle_canvas(self.canvas2, 'Intensity Value', 'Count')

    def rgb_histogram(self,image):
        """
        Generates and plots histograms and cumulative distribution functions (CDFs) for the red, green, and blue color channels of the given image.

        Parameters:
            image : The input image.

        Returns:
            None
        """
        # convert image to gray image
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Calculate the histogram for the red channel (channel[0])


        # Calculate the histogram for the red channel
        red_histogram = self.calculate_histogram(img, 0)

        # Calculate the histogram for the green channel
        green_histogram = self.calculate_histogram(img, 1)

        # Calculate the histogram for the blue channel
        blue_histogram = self.calculate_histogram(img, 2)

        # draw histogram for red color and handel canvas
        self.canvas3.axes.hist(red_histogram, color='red', label='Red Histogram')
        self.handle_canvas(self.canvas3,"Intensity Value",'Count')

        # draw histogram for green color and handel canvas
        self.canvas4.axes.hist(green_histogram, color='green', label='Green Histogram')
        self.handle_canvas(self.canvas4,"Intensity Value",'Count')

        # draw histogram for blue color and handel canvas
        self.canvas5.axes.hist(blue_histogram, color='blue', label='Blue Histogram')
        self.handle_canvas(self.canvas5,"Intensity Value",'Count')

        # Compute cumulative distribution functions (CDFs)
        red_cumulative = self.calculate_cumulative_histogram(red_histogram)
        green_cumulative = self.calculate_cumulative_histogram(green_histogram)
        blue_cumulative = self.calculate_cumulative_histogram(blue_histogram)

        # draw CDF of red color
        self.canvas6.axes.plot(red_cumulative, color='red', label='Red Distribution')
        self.handle_canvas(self.canvas6, "Intensity Value", 'Count')
        # draw CDF of green color
        self.canvas7.axes.plot(green_cumulative, color='green', label='Green Distribution')
        self.handle_canvas(self.canvas7, "Intensity Value", 'Count')
        # draw CDF of blue color
        self.canvas8.axes.plot(blue_cumulative, color='blue', label='Blue Distribution')
        self.handle_canvas(self.canvas8, "Intensity Value", 'Count')


        #draw cumulative curve
        self.canvas9.axes.plot(red_cumulative, color='red', label='Red Cumulative')
        self.canvas9.axes.plot(green_cumulative, color='green', label='Red Cumulative')
        self.canvas9.axes.plot(blue_cumulative, color='blue', label='Red Cumulative')

        self.handle_canvas(self.canvas9, "Intensity Value", 'Count')

    def calculate_histogram(self,image, channel, num_bins=256, intensity_range=(0, 255)):
        """
        Calculates the histogram for a specified color channel of an image.

        Parameters:
            image (numpy.ndarray): The input image.
            channel (int): The color channel to calculate the histogram for (0 for Red, 1 for Green, 2 for Blue).
            num_bins (int): The number of bins in the histogram.
            intensity_range (tuple): The range of intensity values. Default is (0, 255).

        Returns:
            histogram (numpy.ndarray): The computed histogram.
        """
        # Extract the pixel values for a specific color channel
        channel_data = image[:, :, channel]

        # Flatten the channel data to a 1D array
        flattened_channel = channel_data.flatten()

        # Calculate the histogram using NumPy
        histogram, _ = np.histogram(flattened_channel, bins=num_bins, range=intensity_range)

        return histogram
        
    def calculate_cumulative_histogram(self, histogram):
        """
        Calculates the cumulative histogram from a given histogram.

        Parameters:
            histogram (numpy.ndarray): The input histogram.

        Returns:
            cumulative_histogram (list): The cumulative histogram.
        """
        # Initialize the cumulative histogram with the first value of the input histogram
        cumulative_histogram = [histogram[0]]

        # Calculate the cumulative histogram by summing up consecutive values in the input histogram
        for i in range(1, len(histogram)):
            cumulative_histogram.append(cumulative_histogram[-1] + histogram[i])

        # Return the computed cumulative histogram
        return cumulative_histogram

    def clear_canvas(self):
        canvas = [self.canvas1, self.canvas2, self.canvas3, self.canvas4, self.canvas5, self.canvas6,
             self.canvas7, self.canvas8, self.canvas9]
        for canva in canvas:
            canva.axes.clear()

    def handle_canvas(self,canvas,x_labe,y_label):
        # Set labels and legend as needed
        canvas.axes.set_xlabel(x_labe)
        canvas.axes.set_ylabel(y_label)
        canvas.axes.legend()
        canvas.draw()

    # --------------------------------Equalizing--------------------------------
    def equalize_image(self, image):
        """
        This function performs histogram equalization on an input image.

        Parameters:
        image (numpy.ndarray): The input image to be equalized. It can be either a grayscale or a color image.

        The function first checks if the input image is grayscale or color. If it's a color image, it converts the image to HSV color space and only equalizes the V channel (value/brightness). 

        The histogram of the image (or the V channel of the image) is calculated, and then the cumulative distribution function (CDF) of the histogram is computed. The CDF is normalized to the range 0-255, and this normalized CDF is used as a mapping to equalize the image.

        If the input image is color, the equalized V channel is replaced back into the HSV image, and the image is converted back to BGR color space.

        The equalized image is saved to the disk with the filename 'equalized_image.jpg'. The saved image is then loaded in color (BGR format) if it's a color image, or grayscale otherwise.

        Finally, the equalized image is displayed using the `display_filtered_img` method of the class.

        Note: This function uses OpenCV functions such as cv2.calcHist, cv2.cvtColor, cv2.imwrite, and cv2.imread.
        """
        # Check if the image is grayscale or color
        if len(image.shape) == 2:
            # Grayscale image
            img_to_equalize = image
        else:
            # Color image
            # Convert the image to HSV color space
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            img_to_equalize = hsv_img[:,:,2]  # V channel

        # Calculate the histogram
        hist = cv2.calcHist([img_to_equalize], [0], None, [256], [0, 256])

        # Calculate the cumulative distribution function (CDF)
        cdf = hist.cumsum()

        # Normalize the CDF to the range 0-255
        cdf_normalized = cdf * 255 / cdf.max()

        # Use the normalized CDF as a mapping to equalize the image
        equalized_img = cdf_normalized[img_to_equalize.astype('uint8')]

        if len(image.shape) == 2:
            equalized_image = equalized_img
        else:
            # Replace the V channel with the equalized image
            hsv_img[:,:,2] = equalized_img

            # Convert the image back to BGR color space
            equalized_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # Save the equalized image
        cv2.imwrite('equalized_image.jpg', equalized_image)

        # Load the saved image in color (BGR format) if it's a color image, grayscale otherwise
        equalized_image = cv2.imread('equalized_image.jpg', cv2.IMREAD_COLOR if len(image.shape) == 3 else cv2.IMREAD_GRAYSCALE)

        # Display the equalized image
        self.display_filtered_img(equalized_image, self.equalization, self.equalization_scene)

    # --------------------------------Normalizing--------------------------------
    def normalize_image(self, image):
        """
        Normalize given image

        Parameters:
        image : Input image to be normalized.

        Returns:
            None
        
        Description:
            This function takes an input image and normalizes it using the OpenCV library.
            If the input image is in color , it first converts it to grayscale.
            Then, it applies normalization using the cv2.normalize() function, scaling pixel values to the range [0, 255].
            Finally, it displays the normalized image using the display_filtered_img() method.
        """
        # Convert image to grayscale if it's not already
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize the image
        normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # Display the normalized image
        self.display_filtered_img(normalized_image, self.normalization, self.normalization_scene)
        
    # --------------------------------Hybrid Image--------------------------------
    def create_hybrid_image(self):
        """
        This function creates a hybrid image by combining the low-frequency components of one image with the high-frequency components of another image.

        The function first ensures that the two input images are of the same size. If not, it resizes the second image to match the first. It also removes the alpha channel from the images if present.

        A low-pass filter is applied to the first image by blurring it using a Gaussian filter. The cutoff frequency for the low-pass filter is set to 2.5.

        A high-pass filter is applied to the second image by subtracting a blurred version of the image from the original image. The cutoff frequency for the high-pass filter is set to 4.5.

        The low and high-frequency components are then combined to create the hybrid image. The absolute value of the high frequencies is added to ensure the result is still a valid image.

        The hybrid image is saved to the disk with the filename 'hybrid_image.jpg'. The saved image is then loaded in color (BGR format).

        Finally, the hybrid image is displayed using the `display_filtered_img` method of the class.

        Note: This function uses OpenCV functions such as cv2.resize, cv2.imwrite, and cv2.imread.
        """
        print("Creating hybrid image...")
        # Get the first two images from the list
        img1, img2 = self.images_to_mix[:2]

        # Ensure the images are the same size
        if img1.shape != img2.shape:
            print("The images are not of the same size. Resizing the second image to match the first.")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # If the images have an alpha channel, remove it
        if img1.shape[2] == 4:
            img1 = img1[:, :, :3]
        if img2.shape[2] == 4:
            img2 = img2[:, :, :3]

        # Apply a low-pass filter to img1 by blurring it
        cutoff_frequency_1 = 2.5
        gaussian_filter_1 = self.generate_gaussian_filter(shape=(cutoff_frequency_1*4+1,cutoff_frequency_1*4+1), sigma = cutoff_frequency_1)
        low_frequencies = self.apply_filter_to_image(img1, gaussian_filter_1)

        # Apply a high-pass filter to img2 by subtracting a blurred version of img2 from img2 itself
        cutoff_frequency_2 = 4.5
        gaussian_filter_2 = self.generate_gaussian_filter(shape=(cutoff_frequency_2*4+1,cutoff_frequency_2*4+1), sigma = cutoff_frequency_2)
        low_frequencies_2 = self.apply_filter_to_image(img2, gaussian_filter_2)
        high_frequencies = img2 - low_frequencies_2

        # Combine the low frequencies and high frequencies to create the hybrid image
        # We add the absolute value of the high frequencies to ensure the result is still a valid image
        hybrid_image = low_frequencies + high_frequencies

        # Save the hybrid image
        cv2.imwrite('hybrid_image.jpg', hybrid_image)

        # Load the saved image in color (BGR format)
        hybrid_image = cv2.imread('hybrid_image.jpg', cv2.IMREAD_COLOR)

        # Display the hybrid image
        self.display_filtered_img(hybrid_image, self.hybrid_image, self.hybrid_image_scene)

        print("Done creating hybrid image.")

    def apply_filter_to_image(self, image, filter):
        """
        This function applies a given filter to an input image using convolution.

        Parameters:
        image (numpy.ndarray): The input image to which the filter is to be applied. It should be a 3D array representing an RGB image.
        filter (numpy.ndarray): The filter to be applied to the image. It should be a 2D array.

        The function first separates the R, G, and B channels of the image. It then calculates the dimensions of the image and the filter, and computes the amount of padding needed for the convolution operation.

        The R, G, and B channels are then each padded using the 'reflect' mode, which reflects the values at the edge of the array.

        An output image is initialized as a zero array of the same shape as one of the color channels.

        The function then loops over each color channel, and for each pixel in the channel, it applies the filter by performing a convolution operation. The result is a new array for each color channel.

        These new arrays are then stacked together to form a 3D array representing the filtered image. The initial zero array is removed, and the filtered image is returned.

        Note: This function uses numpy functions such as np.pad, np.sum, np.asarray, np.reshape, and np.dstack.
        """
        # Separate the color channels
        red_channel, green_channel, blue_channel = cv2.split(image)
        
        # Calculate dimensions and padding
        image_height, image_width = red_channel.shape
        filter_height, filter_width = filter.shape
        height_padding, width_padding = int((filter_height-1)/2), int((filter_width-1)/2)
        
        # Define padding
        padding = ((height_padding, height_padding), (width_padding, width_padding))
        
        # Pad each color channel
        padded_channels = [np.pad(channel, pad_width=padding, mode='reflect') for channel in [red_channel, green_channel, blue_channel]]
        
        # Initialize output image
        output_image = np.zeros_like(red_channel)
        
        # Apply filter to each color channel
        for channel in padded_channels:
            filtered_channel = []
            
            # Perform convolution
            for row in range(image_height):
                for col in range(image_width):
                    total = np.sum(np.multiply(channel[row:row+filter_height, col:col+filter_width], filter))
                    filtered_channel.append(total)
                    
            # Reshape and stack channels
            filtered_channel = np.asarray(filtered_channel).reshape(image_height, image_width)
            output_image = np.dstack((output_image, filtered_channel))
            
        # Remove initial zero array
        output_image = output_image[:, :, 1:]
        
        return output_image

    def generate_gaussian_filter(self, shape=(3, 3), sigma=0.5):
        """
        This function generates a Gaussian filter with a given shape and standard deviation (sigma).

        Parameters:
        shape (tuple): The shape of the Gaussian filter. It should be a 2D tuple. Default is (3, 3).
        sigma (float): The standard deviation of the Gaussian filter. Default is 0.5.

        The function first calculates the center of the filter. It then generates a grid of (x, y) coordinates that represent the distance from the center.

        The Gaussian function is applied to each (x, y) coordinate to generate the filter. Any values in the filter that are close to zero (less than the machine epsilon times the maximum value in the filter) are set to zero.

        The filter is then normalized so that its sum is 1, and returned.

        Note: This function uses numpy functions such as np.ogrid, np.exp, np.finfo, and np.sum.
        """
        # Calculate the center of the filter
        center_x, center_y = [(dimension - 1) / 2. for dimension in shape]

        # Generate a grid of (x, y) coordinates
        y, x = np.ogrid[-center_x:center_x+1, -center_y:center_y+1]

        # Apply the Gaussian function to each (x, y) coordinate
        gaussian_filter = np.exp(-(x*x + y*y) / (2. * sigma * sigma))

        # Set values close to zero to zero
        gaussian_filter[gaussian_filter < np.finfo(gaussian_filter.dtype).eps * gaussian_filter.max()] = 0

        # Normalize the filter
        sum_of_values = gaussian_filter.sum()
        if sum_of_values != 0:
            gaussian_filter /= sum_of_values

        return gaussian_filter
    
    def low_pass_filter (self, img):
        """
        Apply a low-pass filter to the given image using Fourier Transform.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Filtered image.

        Description:
            This function applies a low-pass filter to the input image in the frequency domain using Fourier Transform.
            It first performs the Fourier Transform on the input image using np.fft.fft2().
            Then, it shifts the zero-frequency component to the center using np.fft.fftshift().
            Next, it creates a filter matrix H, where low frequencies are retained and high frequencies are attenuated.
            The cutoff frequency determines the boundary between low and high frequencies.
            After applying the filter to the frequency-domain representation of the image, it inverse shifts the result
            using np.fft.ifftshift(), applies the inverse Fourier Transform using np.fft.ifft2(), and computes the
            absolute value to obtain the filtered image.
            Finally, it returns the filtered image.
        """
        # Perform Fourier Transform
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        rows, columns = img.shape

        H = np.zeros((rows,columns), dtype=np.float32)
        cutoff_freq = 60

        for u in range(rows):
            for v in range(columns):
                freq = np.sqrt((u-rows/2)**2 + (v-columns/2)**2)
                if freq <= cutoff_freq:
                    H[u,v] = 1
                else:
                    H[u,v] = 0
        Gshift = fshift*H
        G = np.fft.ifftshift(Gshift)
        g =np.abs(np.fft.ifft2(G))
        return g
    
    def high_pass_filter (self, img):
        """
        Apply a high-pass filter to the given image using Fourier Transform.

        Parameters:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Filtered image.

        Description:
            This function applies a high-pass filter to the input image in the frequency domain using Fourier Transform.
            It first performs the Fourier Transform on the input image using np.fft.fft2().
            Then, it shifts the zero-frequency component to the center using np.fft.fftshift().
            Next, it creates a filter matrix H, where low frequencies are attenuated and high frequencies are retained.
            The cutoff frequency determines the boundary between low and high frequencies.
            After applying the filter to the frequency-domain representation of the image, it inverse shifts the result
            using np.fft.ifftshift(), applies the inverse Fourier Transform using np.fft.ifft2(), and computes the
            absolute value to obtain the filtered image.
            Finally, it returns the filtered image.
        """
        # Perform Fourier Transform
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        rows, columns = img.shape

        H = np.zeros((rows,columns), dtype=np.float32)
        cutoff_freq = 60

        for u in range(rows):
            for v in range(columns):
                freq = np.sqrt((u-rows/2)**2 + (v-columns/2)**2)
                if freq <= cutoff_freq:
                    H[u,v] = 0
                else:
                    H[u,v] = 1
        Gshift = fshift*H
        G = np.fft.ifftshift(Gshift)
        g =np.abs(np.fft.ifft2(G))
        return g


    def freq_domain_filter(self):
        """
        Apply frequency domain filtering to the current image.

        Returns:
            None

        Description:
            This method applies frequency domain filtering to the current image based on the selected filter type.
            If a noisy image exists, it uses the noisy image; otherwise, it retrieves the current original image.
            If the original image exists in the scene, it retrieves the selected filter type from the GUI.
            Based on the selected filter type, it performs the following actions:
            - If the selected filter type is "None," it clears the filtered scene and returns.
            - If the selected filter type is "High Pass," it converts the current image to grayscale,
            applies a high-pass filter using the high_pass_filter() method, saves the filtered image to disk,
            and displays the filtered image in the GUI.
            - If the selected filter type is "Low Pass," it converts the current image to grayscale,
            applies a low-pass filter using the low_pass_filter() method, saves the filtered image to disk,
            and displays the filtered image in the GUI.
          """
        # get current original image
        if self.noisy_image is not None:
            curr_image = self.noisy_image
        else:
            curr_image = self.get_current_orignal_img()

        if self.original_scene.items():
            # get current text
            self.current_text = self.freq_filter_combo.currentText()
            print("self.current_text:", self.current_text)
            rgbtogray_img = self.rgbtogray(curr_image) 
            if self.current_text == "None":
                self.filtered_scene.clear()
                return None
            elif self.current_text == "High Pass":
                filtered_image = self.high_pass_filter(rgbtogray_img)
                cv2.imwrite('high_pass.jpg', np.real(filtered_image))
                filtered_image = cv2.imread('high_pass.jpg', cv2.IMREAD_GRAYSCALE)
            elif self.current_text == "Low Pass":
                filtered_image = self.low_pass_filter(rgbtogray_img)
                cv2.imwrite('low_pass.jpg', np.real(filtered_image))
                filtered_image = cv2.imread('low_pass.jpg', cv2.IMREAD_GRAYSCALE)
            
            self.display_filtered_img(filtered_image, self.filtered_image, self.filtered_scene)




def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()