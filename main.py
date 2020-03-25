import glob
import os
import time
import cv2
import dlib
import tkinter as tk
from tkinter import filedialog
import imutils
import pandas as pd
from imageio import imread
from scipy.spatial import distance
from skimage.transform import resize
from keras.models import load_model
from tqdm import tqdm
import filetype as ft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from align import AlignDlib
from model import create_model
from keras.callbacks import TensorBoard
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_DIR = "D:/data/images"

# image_extentions = ['.jpg', '.jpx', '.png', '.gif', '.webp', '.cr2', '.tif', '.bmp', '.jxr', '.pxd', '.ico', '.heic']
hog_face_detector = dlib.get_frontal_face_detector()

# cnn_face_detector = dlib.cnn_face_detection_model_v1('weights/mmod_human_face_detector.dat')

model = None
alignment = None
# label2idx = None
threshold = 1
train_embs = None

df_train = None  # 3DIM
train_paths = None
nb_classes = None

match_distences = []
unmatch_distences = []


def create_squential_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


# INITIALIZE MODELS
def init_sequential_model():
    global model, alignment, df_train, train_paths

    model = create_squential_model()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    NAME = 'CNN-{}'.format(int(time.time()))
    tboard_log_dir = os.path.join("logs", NAME)
    tensorboard_callback = TensorBoard(log_dir=tboard_log_dir, histogram_freq=1)

    model.fit(x=x_train,
              y=y_train,
              epochs=5,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])

    alignment = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')


def init_model():
    global model, alignment, df_train
    model = create_model()
    model.summary()
    model.load_weights("weights/nn4.small2.v1.h5")
    alignment = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')


# LOAD DATA - TRAINING INFORMATION


def load_data(path):
    global train_paths, df_train, nb_classes

    df_train = pd.DataFrame(columns=['image', 'label', 'name'])

    train_paths = glob.glob(path)

    nb_classes = len(train_paths)

    print(train_paths)

    # tqgdm de the hien qua trinh xu ly
    for i, train_path in tqdm(enumerate(train_paths)):
        name = train_path.split("\\")[-1]
        images = glob.glob(train_path + "/*")
        for image in images:
            df_train.loc[len(df_train)] = [image, i, name]

    print(df_train.head())


# PRE_PROCESSING

# Các bước load ảnh và chuẩn hóa trước khi cho vào mạng.
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def align_face(face):
    # print(img.shape)
    (h, w, c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    # print(bb)
    return alignment.align(96, face, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def load_and_align_images(file_paths):
    aligned_images = []
    for filepath in file_paths:
        # print(file_paths)
        img = cv2.imread(filepath)
        aligned = align_face(img)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)

    return np.array(aligned_images)


# Hàm tính embedding
def calc_embs(file_paths, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(file_paths), batch_size)):
        aligned_images = load_and_align_images(file_paths[start:start + batch_size])
        pd.append(model.predict_on_batch(np.squeeze(aligned_images)))
    embs = l2_normalize(np.concatenate(pd))
    # embs = np.array(pd)
    return np.array(embs)


def align_faces(faces):
    aligned_images = []
    for face in faces:
        # print(face.shape)
        aligned = align_face(face)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)

    return aligned_images


def calc_emb_test(faces):
    pd = []
    aligned_faces = align_faces(faces)
    if len(faces) == 1:
        pd.append(model.predict_on_batch(aligned_faces))
    elif len(faces) > 1:
        pd.append(model.predict_on_batch(np.squeeze(aligned_faces)))
    embs = l2_normalize(np.concatenate(pd))
    # embs = np.array(pd)
    return np.array(embs)


# TRAINING

def training():
    global label2idx, df_train, train_embs

    label2idx = []

    for i in tqdm(range(len(train_paths))):
        label2idx.append(np.asarray(df_train[df_train.label == i].index))

    train_embs = calc_embs(df_train.image)

    np.save("train_embs.npy", train_embs)

    train_embs = np.concatenate(train_embs)


# ANALYSING

def analysing():
    global label2idx, nb_classes, train_embs, match_distances, unmatch_distances

    match_distances = []

    for i in range(nb_classes):
        ids = label2idx[i]
        distances = []
        for j in range(len(ids) - 1):
            for k in range(j + 1, len(ids)):
                distances.append(distance.euclidean(train_embs[ids[j]].reshape(-1), train_embs[ids[k]].reshape(-1)))
        match_distances.extend(distances)

    unmatch_distances = []
    for i in range(nb_classes):
        ids = label2idx[i]
        distances = []
        for j in range(10):
            idx = np.random.randint(train_embs.shape[0])
            while idx in label2idx[i]:
                idx = np.random.randint(train_embs.shape[0])
            distances.append(distance.euclidean(train_embs[ids[np.random.randint(len(ids))]].reshape(-1),
                                                train_embs[idx].reshape(-1)))
        unmatch_distances.extend(distances)
        # if matplotlib is delete, program hang after plt.show

        # matplotlib.interactive(True)

    # _, _, _ = plt.hist(match_distances, bins=100)
    # _, _, _ = plt.hist(unmatch_distances, bins=100, fc=(1, 0, 0, 0.5))
    # print(match_distances)
    # print(unmatch_distances)
    # plt.show()


# TEST

def test(paths):  # /*.jpg test_image/*.jpg
    global label2idx, threshold, train_embs
    print(time.time())
    test_paths = glob.glob(paths)

    for path in test_paths:
        print(time.time())
        test_image = cv2.imread(path)
        show_image = test_image.copy()

        faceRects = hog_face_detector(test_image, 0)

        faces = []

        for faceRect in faceRects:
            print(time.time())
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            face = test_image[y1:y2, x1:x2]
            faces.append(face)

        print("len(faces) = {0}".format(len(faces)))
        if len(faces) == 0:
            print("No face detected!")
            continue
        else:
            test_embs = calc_emb_test(faces)

        test_embs = np.concatenate(test_embs)

        people = []
        for i in range(test_embs.shape[0]):
            print(time.time())
            distances = []
            for j in range(len(train_paths)):
                print(time.time())
                distances.append(np.min(
                    [distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)) for k in label2idx[j]]))
                # for k in label2idx[j]:
                # print(distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)))
            if np.min(distances) > threshold:
                people.append("unknown")
            else:
                res = np.argsort(distances)[:1]
                people.append(res)

        names = []
        title = ""
        for p in people:
            print(time.time())
            if p == "unknown":
                name = "unknown"
            else:
                name = df_train[(df_train['label'] == p[0])].name.iloc[0]
            names.append(name)
            title = title + name + " "

        for i, faceRect in enumerate(faceRects):
            print(time.time())
            x1 = faceRect.left()
            y1 = faceRect.top()
            x2 = faceRect.right()
            y2 = faceRect.bottom()
            cv2.rectangle(show_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(show_image, names[i], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        show_image = imutils.resize(show_image, width=720)
        cv2.imshow("Result", show_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Nhan dan khuon mat thong qua thiet bi camera
def face_recognition_camera_HOG(device):
    global hog_face_detector
    # hog_face_detector = dlib.get_frontal_face_detector()
    if hog_face_detector in globals() or locals():
        try:
            cap = cv2.VideoCapture(device)
            while True:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                ret, frame = cap.read()
                # f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
                # cv2.imshow('Frame', gray)
                faces_hog = hog_face_detector(f, 0)
                # faces_hog = cnn_face_detector(f,1)
                for face in faces_hog:
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y

                    cv2.rectangle(f, (x, y), (w + x, h + y), (0, 255, 0), 2)

                cv2.imshow('Frame', f)
            cap.release()
            cv2.destroyAllWindows()
        except:
            print('An error occurred')
    else:
        print('Hog does not exists')
        return


# Nhan dan khuon mat thong qua hinh anh
def face_recognition_image_HOG(path):
    if path is None:
        return
    else:
        name = os.path.basename(path)
        img = cv2.imread(path)
        print(name)
        # Thực hiện xác định bằng HOG và SVM
        start = time.time()
        faces_hog = hog_face_detector(img, 1)
        # faces_hog = cnn_face_detector(img, 1)
        end = time.time()
        print("HOG + SVM Executing time: " + str(end - start))

        # Vẽ một dường màu xanh bao quanh các khuôn mặt được xác định ra bởi HOG và SVM
        for face in faces_hog:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y

            cv2.rectangle(img, (x, y), (w + x, h + y), (0, 255, 0), 2)
        # for face in faces_hog:
        #     x = face.rect.left()
        #     y = face.rect.top()
        #     w = face.rect.right() - x
        #     h = face.rect.bottom() - y
        #
        #     cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 2)
        cv2.imshow(name, img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC to exit
            cv2.destroyAllWindows()
            return
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite('picure.jpg', img)


# Duyet windows tim duong dan den thu muc
root = tk.Tk()
root.withdraw()


def browse_windows_file():
    file_path = filedialog.askopenfilename()
    print(file_path)

    try:
        kind = ft.guess(file_path)
        if ft.is_extension_supported(kind.extension) is False:
            print("This extension isn't supported")
            return None
        else:
            # root.destroy()
            return file_path
    except AttributeError as ae:
        print(str(ae))


def browse_windows_folder():
    path = filedialog.askdirectory()
    print(path)

    try:
        if os.path.isdir(path) is False:
            print("This extension isn't supported")
            return None
        else:
            # root.destroy()
            return path
    except AttributeError as ae:
        print(str(ae))


def main():
    # face_recognition_camera_HOG(0)
    # face_recognition_image_HOG('Abdoulaye_Wade_0004.jpg')

    # face_recognition_image_HOG(file_path)
    global threshold
    threshold = 1

    # path = DATA_DIR + "/*"
    path = browse_windows_folder() + "/*"
    file_path = browse_windows_file()
    print("INIT")
    init_model()
    # init_sequential_model()
    print("LOAD")
    load_data(path)
    print("TRAIN")
    training()
    print("ANALYSING")
    analysing()
    print("TEST")
    test(file_path)


if __name__ == "__main__":
    main()
