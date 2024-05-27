import random
from PIL import Image
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

from face_recognition import FaceRecongnition

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

        #rename taps
        self.tabWidget.setTabText(0, "Face Detection")
        self.tabWidget.setTabText(1, "Face Recognition")
        self.tabWidget.setTabText(2, "ROC")
        self.handle_ui()
        self.detection_browseButton.clicked.connect(lambda: self.browse_image(self.input_scene_1))  # Connect the clicked signal to browse_image function
        self.recognition_browseButton.clicked.connect(lambda: self.browse_image(self.input_scene_2))  # Connect the clicked signal to browse_image function
        self.detection_applyButton.clicked.connect(self.handle_apply_button)
        self.recognition_button.clicked.connect(self.face_recognition)
        self.draw_roc_button.clicked.connect(self.draw_roc_curve)

        self.canvas1, self.layout1 = self.create_canvas_layout(self.graphicsView_5)

        # Set the layout for the respective widgets
        self.graphicsView_5.setLayout(self.layout1)

    
    def handle_ui(self):
        """
        Handle the UI elements.
        :return: None
        """
        # Create a QGraphicsScene
        self.input_scene_1 = QtWidgets.QGraphicsScene()
        self.input_graphicsView_1.setScene(self.input_scene_1)
        self.input_scene_2 = QtWidgets.QGraphicsScene()
        self.input_graphicsView_2.setScene(self.input_scene_2)
        self.output_scene_1 = QtWidgets.QGraphicsScene()
        self.output_graphicsView_1.setScene(self.output_scene_1)
        self.output_scene_2 = QtWidgets.QGraphicsScene()
        self.output_graphicsView_face_recognition.setScene(self.output_scene_2)



    def browse_image(self, scene):
        """
        Browse an image file and display it in the given scene.
        :param scene: The QGraphicsScene to display the image in.
        :return: None
        """
        # Open a file dialog to select an image file
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("All Files (*)")  # Include all files
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.image_path = file_path

            # Open the image file using PIL
            pil_image = Image.open(file_path)

            # Convert the PIL Image to a numpy array
            image = np.array(pil_image)

            # If the image is grayscale (such as a .pgm file), then cvtColor will throw an error. 
            # So we only convert to RGB if the image has more than one channel.
            if len(image.shape) > 2:
                # Convert the BGR image to RGB
                self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # If the image is grayscale, we just copy it to image_rgb
                self.image_rgb = image.copy()

            self.current_img = image

            # Convert the image to QImage
            height, width = self.image_rgb.shape[:2]
            bytes_per_line = width
            if len(self.image_rgb.shape) > 2:
                bytes_per_line *= 3
                q_image = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            else:
                q_image = QImage(self.image_rgb.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

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
        """
        Displays a QPixmap in a QGraphicsView within a specified scene.

        Args:
            pixmap (QPixmap): The QPixmap to display.
            scene (QGraphicsScene): The QGraphicsScene to display the QPixmap in.

        Returns:
            None
        """
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

    def handle_apply_button(self):
        """
        Handles the apply button click event.

        Returns:
            None
        """
        if self.tabWidget.currentIndex() == 0:
            self.detect_face()

    # --------------------------------Face Detection--------------------------------  
    def detect_face(self):
        """
        Detects faces in an image using a Haar cascade classifier.
        Draws a rectangle around the detected faces.
        Displays the filtered image in the output scene.
        Returns:
            None
        """
        # Check the number of channels in the image
        if len(self.current_img.shape) == 3 and self.current_img.shape[2] == 3:
            img_rgb = self.current_img
        else:
            # Convert grayscale image to RGB
            img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_GRAY2RGB)

        # Load the cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(self.current_img, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)  # BGR color for the rectangle is now red
        
        img_BGR = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        self.display_filtered_img(img_BGR, self.output_graphicsView_1, self.output_scene_1)

    # --------------------------------Face Recognition--------------------------------    
    def face_recognition(self):
        """
        This method recognizes a face in an image and displays the result in a graphical user interface. If the face is not recognized, it displays an error message.

        Parameters:
        self: The instance of the class. This method uses the `image_path`, `output_graphicsView_face_recognition`, and `output_scene_2` attributes of the class instance.

        Returns:
        None. This method does not return anything. It modifies the instance variables `output_graphicsView_face_recognition` and `output_scene_2` in-place.
        """
        test = FaceRecongnition("train")
        result = test.fit(self.image_path)
        print(result)
        if result[0] == "unknown Face":
            print("unknown Face")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Unknown Face")
            msg.setWindowTitle("Error")
            # Adjust the width of the message box to fit the text
            msg.layout().setSizeConstraint(QLayout.SetFixedSize)
            msg.exec_()
        else:
            print("Face Is Recognized")
            path = "train/" + result[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read the image data
            self.display_filtered_img(img, self.output_graphicsView_face_recognition, self.output_scene_2)

    # --------------------------------ROC--------------------------------
    def draw_roc_curve(self):
        test = FaceRecongnition("train")
        result,auc_result,accuracy=test.perform_roc_analysis()
        self.plot(result)
        self.label_2.setText("AUC: "+ str(auc_result))
        self.label_3.setText("Accuracy: "+ str(accuracy))

    def plot(self, result):
        y = [result[0], result[2], result[4], result[6], result[8]]
        x = [result[1], result[3], result[5], result[7], result[9]]
        self.canvas1.axes.clear()
        self.canvas1.axes.set_xlabel("False Positive Rate")
        self.canvas1.axes.set_ylabel("True Positive Rate")
        self.canvas1.axes.plot(x,y, color='blue', label='ROC')
        self.canvas1.axes.legend()
        self.canvas1.draw()

    def create_canvas_layout(self, graph_widget):
        canvas = MplCanvas(graph_widget, width=20, height=5, dpi=100)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(canvas)
        return canvas, layout

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()