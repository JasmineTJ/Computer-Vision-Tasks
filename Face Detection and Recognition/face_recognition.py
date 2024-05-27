import os

from sklearn.metrics import roc_auc_score, auc
from sklearn.preprocessing import normalize
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt



class FaceRecongnition:
    def __init__(self, dir_to_input: str) -> None:
        self.files_list = 0  # list of sample's files names
        self.dir_to_input = dir_to_input  # dir to the sample
        self.face_matrix = np.array([])
        self.eigen_faces = np.array([])
        
    def get_files_list(self):
        """
        Retrieves all files in a directory and stores them into a list.

        Parameters:
        self: The instance of the class.

        Returns:
        None. This method does not return anything. It modifies the instance variable `self.files_list` in-place.
        """
        # Containing a list of filenames in the directory specified by
        self.files_list = os.listdir(self.dir_to_input)
        print("length of files: ", len(self.files_list))
        
    def create_face_matrix(self):
        """
        This method reads all image files specified in `self.files_list`, resizes each image to 100x100 pixels, 
        reshapes each image into a vector, and stores these vectors in `self.face_matrix`.

        Parameters:
        self: The instance of the class. This method uses the `files_list` and `dir_to_input` attributes of the class instance.

        Returns:
        None. This method does not return anything. It modifies the instance variable `self.face_matrix` in-place.
        """
        face_matrix = []
        for i in range(len(self.files_list)):
            # read img as greyscale
            img = cv.imread(self.dir_to_input + "/" + self.files_list[i], cv.IMREAD_GRAYSCALE)
            # resize img
            img = cv.resize(img, (100, 100))
            # convert img into vector
            num_of_dimensions = img.shape[0] * img.shape[1]
            # Reshapes the image into a one-dimensional array (or vector) with a length of num_of_dimensions
            img = np.reshape(img, num_of_dimensions)
            print("image content: ", img)
            print("image shape: ", img.shape)
            
            # add to face matrix
            face_matrix.append(img)
        self.face_matrix = np.array(face_matrix)
        print("face matrix dimension: ", self.face_matrix.shape)
        
    def get_mean_matrix(self) -> np.ndarray:
        """
        This method calculates the mean of the face matrix and subtracts this mean from all samples in the face matrix, resulting in a mean array.

        Parameters:
        self: The instance of the class. This method uses the `face_matrix` attribute of the class instance.

        Returns:
        np.ndarray: A mean numpy array. Each row of the array corresponds to a sample (image), and each column corresponds to a feature (pixel).
        """
        # Calculates the mean of each column in self.face_matrix
        self.mean_sample = np.mean(self.face_matrix, axis=0)
        return np.subtract(self.face_matrix, self.mean_sample)  # mean array
    
    def get_covariance_matrix(self):
        """
        This method calculates the covariance matrix of the mean array, computes its eigenvalues and eigenvectors, 
        sorts the eigenvectors in descending order of their corresponding eigenvalues, calculates a linear combination 
        of the mean array and the sorted eigenvectors, normalizes the result, and assigns it to `self.eigen_faces`.

        Parameters:
        self: The instance of the class. This method uses the `mean_arr` and `files_list` attributes of the class instance.

        Returns:
        None. This method does not return anything. It modifies the instance variable `self.eigen_faces` in-place.
        """
        cov = (self.mean_arr.dot(self.mean_arr.T)) / (len(self.files_list) - 1)

        # get eigenvalues and eigenvectors of cov
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        print("Eigen Values", eigenvalues)
        print("Eigen Vectors", eigenvectors)
        
        # Order eigenvalues by index desendingly
        idx = eigenvalues.argsort()[::-1]
        print("indexes: ", idx)
        # sort eigenvectors according to the sorted indices of the eigenvalues
        eigenvectors = eigenvectors[:, idx]
        # linear combination of each column of mean_mat
        eigenvectors_c = self.mean_arr.T @ eigenvectors
        # normalize the eigenvectors
        # normalize only accepts matrix with n_samples, n_feature. Hence the transpose.
        self.eigen_faces = normalize(eigenvectors_c.T, axis=1)
        print("EigenFaces", self.eigen_faces)
        print("eigen faces shape", self.eigen_faces.shape) # 64 x 10000
        
    def detect_face(self, img_path):
        """
        This method reads an image file, resizes it, subtracts the mean sample, calculates the dot product of the eigen faces 
        and the mean subtracted image, checks if a face is detected in the image based on a threshold, and returns the vector 
        of the mean subtracted image.

        Parameters:
        self: The instance of the class. This method uses the `eigen_faces` and `mean_sample` attributes of the class instance.
        img_path: str. The path to the image file that will be processed.

        Returns:
        np.ndarray: The vector of the mean subtracted image.
        """
        found_flag = 0  # Flag to check if face is found in the dataset
        # testing image
        test_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        # resize the testing image. cv2 resize by width and height.
        test_img = cv.resize(test_img, (100, 100))
        # subtract the mean
        mean_subtracted_test_img = np.reshape(test_img, (100 * 100)) - self.mean_sample
        # print(mean_subtracted_test_img.shape)
        # the vector that represents the image with respect to the eigenfaces. (projecting the image onto the eigenfaces by doing it in the original image space)
        vector_of_mean_subtracted_test_img = self.eigen_faces.dot(mean_subtracted_test_img)
        # print(vector_of_mean_subtracted_test_img.shape)
        # chosen threshold for face detection
        alpha_1 = 3000
        # n^2 vector of the new face image represented as the linear combination of the chosen eigenfaces (projecting the image onto the eigenfaces by doing it in the reduced-dimensionality face space (due to the transpose))
        # 90 % of dataset is number of chosen eigenfaces
        projected_new_img_vector = self.eigen_faces.T @ vector_of_mean_subtracted_test_img
        # distance between the original face image vector and the projected vector.
        diff = mean_subtracted_test_img - projected_new_img_vector
        # calculates the Euclidean distance between the original face image vector and the projected vector
        # It’s a measure of how different the original image is from the reconstructed image
        beta = math.sqrt(diff.dot(diff))

        if beta < alpha_1:
            print(f"Face detected in the image!, beta = {beta}")
            found_flag = 1
        else:
            print(f"No face detected in the image!, beta = {beta} ")
        return vector_of_mean_subtracted_test_img
    
    def face_recognition(self, vector_of_mean_subtracted_test_img, percent):
        """
        This method recognizes a face in an image by comparing it to a dataset of known faces.

        Parameters:
        self: The instance of the class. This method uses the `files_list`, `eigen_faces`, and `mean_arr` attributes of the class instance.
        vector_of_mean_subtracted_test_img: np.ndarray. The vector representation of the image with the mean subtracted.
        percent: float. The percentage of the dataset to consider for face recognition.

        Returns:
        tuple: A tuple containing the label of the recognized face and a flag indicating whether a face was recognized. If a face was recognized, the label is the filename of the recognized face and the flag is 1. If no face was recognized, the label is the string "unknown Face" and the flag is 0.
        """
        threshold = 3000
        label = 0
        #  start distance with 0
        smallest_distance = 0
        #  iterate over all image vectors until fit input image
        for i in range(len(self.files_list)):
            # projecting each image in face space (The result is a vector that represents the i-th image in the face space)
            Edp = self.eigen_faces.dot(self.mean_arr[i])
            # print(Edp.shape)
            # calculating euclidean distance between vectors
            differnce = vector_of_mean_subtracted_test_img[:int(percent*len(self.files_list))] - Edp[:int(percent*len(self.files_list))]
            euclidean_distances = math.sqrt(differnce.dot(differnce))
            # get smallest distance
            if smallest_distance == 0:
                smallest_distance = euclidean_distances
                label = i
            if smallest_distance > euclidean_distances:
                smallest_distance = euclidean_distances
                label = i
        # comparing smallest distance with threshold
        if smallest_distance < threshold:
            k = 1
            # print("the input image fit :", self.FilesList[label])
            return self.files_list[label], k
        else:
            k = 0
            # print("unknown Face")
            return "unknown Face", k
    
    def fit(self, test_img):
        self.get_files_list()
        self.create_face_matrix()
        self.mean_arr = self.get_mean_matrix()
        self.get_covariance_matrix()
        vector_of_mean_subtracted_test_img = self.detect_face(test_img)
        result = self.face_recognition(vector_of_mean_subtracted_test_img, 0.9)
        return result

    def roc(self, folder_path, threshold):
        img_list = os.listdir(folder_path)
        output_images = []
        labels = []
        for img in img_list:
            test_img = cv.imread(folder_path + '/' + img, cv.IMREAD_GRAYSCALE)
            # resize the testing image. cv2 resize by width and height.
            test_img = cv.resize(test_img, (100, 100))
            # subtract the mean
            mean_subtracted_test_img = np.reshape(test_img, (100 * 100)) - self.mean_sample
            # the vector that represents the image with respect to the eigenfaces.
            vector_of_mean_subtracted_test_img = self.eigen_faces.dot(mean_subtracted_test_img)
            result, label = self.face_recognition(vector_of_mean_subtracted_test_img, threshold)
            output_images.append(result)
            labels.append(label)

        roc,accuracy = self.compare(img_list, output_images, labels)
        return roc,accuracy

    def compare(self, x, y, label):
        roc = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for l in range(20):
            txt = x[l]
            I = txt.split('_')
            txt2 = y[l]
            O = txt2.split('_')
            if I[0] in ['yaleB01', 'yaleB02', 'yaleB03', 'yaleB04', 'yaleB05', 'yaleB06', 'yaleB07', 'yaleB08']:
                if I[0] == O[0]:
                    print("I[0],O[0]",I[0],O[0])
                    tp = tp + 1
                    print("tp")
                else:
                    fn = fn + 1
                    print("fn")
            else:
                if label[l] == 1:
                    fp = fp + 1
                    print("fp")
                else:
                    tn = tn + 1
                    print("tn")
        print(tp, fn, fp, tn)
        tpr = tp / (tp + fn)
        fpr = tn / (tn + fp)
        accuracy=(tp+tn)/(tp+tn+fp+fn)
        print("accuracy",accuracy)
        # roc_auc = auc(fpr, tpr)
        print("fpr,tpr",fpr,tpr)
        roc.extend([tpr, fpr])
        return roc,accuracy

    def perform_roc_analysis(self):
        thresholds = [0.01, 0.02, 0.05, 0.5, 0.9]
        result = []
        self.get_files_list()
        self.create_face_matrix()
        self.mean_arr = self.get_mean_matrix()
        self.get_covariance_matrix()
        for threshold in thresholds:
            roc ,accuracy= self.roc("test", threshold)
            print("roccccccccc",roc)
            result.append(roc)
        results = np.array(result)
        roc_data = np.array(result)
        auc_result=self.report_performance(roc_data)
        return results.flatten(),auc_result,accuracy

    def report_performance(self, roc_data):
        """
        Report the performance metrics including AUC.

        Parameters:
        roc_data: np.ndarray. ROC data containing True Positive Rate (TPR) and False Positive Rate (FPR) values.

        Returns:
        float: Area under the ROC curve (AUC).
        """
        tpr_values = roc_data[:, 0]
        fpr_values = roc_data[:, 1]

        # Calculate AUC using trapezoidal rule
        auc_roc = auc(fpr_values,tpr_values )



        print("Area under the ROC curve (AUC):", auc_roc)


        return auc_roc







