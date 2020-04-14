import sys
import dlib
import numpy as np
import cv2

import time
import dlib
import cv2
import matplotlib.pyplot as plt

# Display one image
from imutils import face_utils


def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display two images
def display(a, b, title1="Hog", title2="CNN"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


path = r'D:\Data\face_rec\faces\obama.jpg'
# path = r'D:\Data\face_rec\faces\happy-people-1050x600.jpg'
# Đọc ảnh đầu vào
image_1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_2 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
# Khai báo việc sử dụng các hàm của dlib
hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1('model/mmod_human_face_detector.dat')

# # Thực hiện xác định bằng HOG và SVM
# start = time.time()
# faces_hog = hog_face_detector(image_1, 1)
# end = time.time()
# print("Hog + SVM Execution time: " + str(end - start))
#
# # Vẽ một đường bao màu xanh lá xung quanh các khuôn mặt được xác định ra bởi HOG + SVM
# for face in faces_hog:
#     x = face.left()
#     y = face.top()
#     w = face.right() - x
#     h = face.bottom() - y
#
#     cv2.rectangle(image_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# # Thực hiện xác định bằng CNN
# start = time.time()
# faces_cnn = cnn_face_detector(image_2, 1)
# end = time.time()
# print("CNN Execution time: " + str(end - start))
#
# # Vẽ một đường bao đỏ xung quanh các khuôn mặt được xác định bởi CNN
# for face in faces_cnn:
#     x = face.rect.left()
#     y = face.rect.top()
#     w = face.rect.right() - x
#     h = face.rect.bottom() - y
#
#     cv2.rectangle(image_2, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# display(image_1, image_2)

p = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
image_1 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for (x, y) in shape:
        cv2.circle(image_1, (x, y), 2, (0, 255, 0), -1)

# Show the image
display_one(image_1, "Image")
