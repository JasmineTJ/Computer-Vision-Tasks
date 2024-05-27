import numpy as np
import cv2
import sift
from matplotlib import pyplot as plt

def sift_image(radio_button_name):

    MIN_MATCH_COUNT = 10

    img1 = cv2.imread('data/box.jpg', 0)           # queryImage
    img2 = cv2.imread('data/box_in_scene.jpg', 0)  # trainImage

    # Compute SIFT keypoints and descriptors
    kp1, des1 = sift.sift(img1)
    kp2, des2 = sift.sift(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = img1.shape
        pts = np.float32([[0, 0],
                        [0, h - 1],
                        [w - 1, h - 1],
                        [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = img1.shape
        h2, w2 = img2.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = img1
            newimg[:h2, w1:w1 + w2, i] = img2

        if radio_button_name == "matching_radioButton":
            # Draw SIFT keypoint matches
            for m in good:
                pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
                pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
                cv2.line(newimg, pt1, pt2, (255, 0, 0))
            cv2.imwrite('sift.jpeg', np.real(newimg))

        elif radio_button_name == "keypoints_radioButton":
            # Draw keypoints on image1
            img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('keypoints_img1.jpg', img1_kp)

            # Draw keypoints on image2
            img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('keypoints_img2.jpg', img2_kp)
            
            # Load the keypoints images
            img1_kp = cv2.imread('keypoints_img1.jpg')
            img2_kp = cv2.imread('keypoints_img2.jpg')

            # Ensure both images have the same width
            min_width = min(img1_kp.shape[1], img2_kp.shape[1])
            img1_kp = img1_kp[:, :min_width]
            img2_kp = img2_kp[:, :min_width]

            # Combine the images vertically
            combined_image = cv2.vconcat([img1_kp, img2_kp])

            # Save the combined image
            cv2.imwrite('combined_keypoints_vertical.jpg', combined_image)

        else:
            print("No radio button is chosen")    