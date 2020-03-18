import glob
import os
import time
import cv2
import dlib
import tkinter as tk
from tkinter import filedialog
import imutils
import pandas as pd
from scipy.spatial import distance
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
label2idx = None
threshold = 1
train_embs = None

df_train = pd.DataFrame(columns=['image', 'label', 'name'])  # 3DIM
train_paths = None
nb_classes = None

match_distences = []
unmatch_distences = []


# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=3)
#
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)
# model.save('epic_reader_num.model')
# new_model = tf.keras.models.load_model('epic_reader_num.model')
# predictions =  new_model.predict([x_test])
# print(predictions)
# print(np.argmax(predictions[0]))
#
# plt.imshow(x_test[0])
# plt.show()


# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()


# face_cascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_eye.xml')
#
# img = cv2.imread('people.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x, y, w, h) in faces:
#     img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = img[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Khai báo việc sử dụng các hàm của dlib


# # Xác định bằng CNN
# start = time.time()
# faces_cnn = cnn_face_detector(img, 1)
# end = time.time()
# print("CNN Executing time: ", str(end - start))
# # Vẽ một đường màu đỏ bao quanh các khuôn mặt được xác định bỏi CNN
# for face in faces_cnn:
#     x = face.rect.left()
#     y = face.rect.top()
#     w = face.rect.right() - x
#     h = face.rect.bottom() - y
#
#     cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 2)
#

# HUẤN LUYỆN MÔ HÌNH
# MODEL CONVNET

# def convnet_model_():
#     vgg_model = applications.VGG16(weights=None, include_top=False, input_shape=(221, 221, 3))
#     x = vgg_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(4096, activation='relu')(x)
#     x = Dropout(0.6)(x)
#     x = Dense(4096, activation='relu')(x)
#     x = Dropout(0.6)(x)
#     x = Lambda(lambda x_: K.l2_normalize(x, axis=1))(x)
#     # x = Lambda(K.l2_normalize)(x)
#     convnet_model = Model(inputs=vgg_model.input, outputs=x)
#     return convnet_model
#
#
# def deep_rank_model():
#     convnet_model = convnet_model_()
#
#     first_input = Input(shape=(221, 221, 3))
#     first_conv = Conv2D(96, kernel_size=(8, 8), strides=(16, 16), padding='same')(first_input)
#     first_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(first_conv)
#     first_max = Flatten()(first_max)
#     first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)
#
#     second_input = Input(shape=(221, 221, 3))
#     second_conv = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(second_input)
#     second_max = MaxPool2D(pool_size=(7, 7), strides=(4, 4), padding='same')(second_conv)
#     second_max = Flatten()(second_max)
#     second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)
#
#     merge_one = concatenate([first_max, second_max])
#     merge_two = concatenate([merge_one, convnet_model.output])
#     emb = Dense(4096)(merge_two)
#     emb = Dense(128)(emb)
#     l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)
#
#     final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)
#
#     return final_model

# LOAD MODEL

# deep_rank_model = deep_rank_model()
# deep_rank_model.load_weights('/home/pham.hoang.anh/prj/face_detect/triplet_weight.hdf5')

# Load all vector embedding LFW of my model
# with open('/home/pham.hoang.anh/prj/face_detect/embs128.pkl', 'rb') as f:
#     embs128 = pickle.load(f)
# with open(
#         '/home/pham.hoang.anh/prj/face_detect/visualize/128D-Facenet-LFW-Embedding-Visualisation/oss_data/LFW_128_HA_labels.tsv',
#         'r') as f:
#     names = f.readlines()

# HÀM TRIPLET LOSS
# batch_size = 24
#
# _EPSILON = K.epsilon()
#
#
# def _loss_tensor(y_true, y_pred):
#     y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
#     loss = 0.
#     g = 1.
#     for i in range(0, batch_size, 3):
#         try:
#             q_embedding = y_pred[i]
#             p_embedding = y_pred[i + 1]
#             n_embedding = y_pred[i + 2]
#             D_q_p = K.sqrt(K.sum((q_embedding - p_embedding) ** 2))
#             D_q_n = K.sqrt(K.sum((q_embedding - n_embedding) ** 2))
#             loss = loss + g + D_q_p - D_q_n
#         except:
#             continue
#     loss = loss / batch_size * 3
#     return K.maximum(loss, 0)
#
#
# deep_rank_model.compile(loss=_loss_tensor, optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))
#
#
# def image_batch_generator(images, labels, batch_size):
#     labels = np.array(labels)
#     while True:
#         batch_paths = np.random.choice(a=len(images), size=batch_size // 3)
#         input_1 = []
#
#         for i in batch_paths:
#             pos = np.where(labels == labels[i])[0]
#             neg = np.where(labels != labels[i])[0]
#
#             j = np.random.choice(pos)
#             while j == i:
#                 j = np.random.choice(pos)
#
#             k = np.random.choice(neg)
#             while k == i:
#                 k = np.random.choice(neg)
#
#             input_1.append(images[i])
#             input_1.append(images[j])
#             input_1.append(images[k])
#
#         input_1 = np.array(input_1)
#         input = [input_1, input_1, input_1]
#         yield input, np.zeros((batch_size,))

# deep_rank_model.fit_generator(generator=image_batch_generator(X, y, batch_size),
#                               steps_per_epoch=len(X) // batch_size,
#                               epochs=2000,
#                               verbose=1,
#                               callbacks=callbacks_list)

# nn4_small2 = create_model()
# nn4_small2.load_weights('weights/nn4.small2.v1.h5')

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


def calc_embs(file_paths, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(file_paths), batch_size)):
        aligned_images = load_and_align_images(file_paths[start:start + batch_size])
        pd.append(model.predict_on_batch(np.squeeze(aligned_images)))
    # embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)
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
    # embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)
    return np.array(embs)

    # df_train.to_pickle('images/data.pkl')

    # plt.imread(data)
    # plt.show()
    # LUU TRU DU LIEU
    # pickle_out = open("x.pickle", "wb")
    # pickle.dump("x", pickle_out)
    # pickle_out.close()
    #
    # pickle_out = open("y.pickle", "wb")
    # pickle.dump("y", pickle_out)
    # pickle_out.close()
    #
    # pickle_in = open("x.pickle", "rb")
    # x = pickle.load(pickle_in)

    # Nhan dien khuon mat va cat no ra de xu ly nhanh hon
    # for img_path in df_train.image:
    #     try:
    #         print(img_path)
    #         image = imread(img_path)
    #         faceRects = hog_face_detector(image, 0)
    #         faceRect = faceRects[0]
    #         if faceRect is None:
    #             continue
    #
    #         x1 = faceRect.left()
    #         y1 = faceRect.top()
    #         x2 = faceRect.right()
    #         y2 = faceRect.bottom()
    #
    #         face = image[y1:y2, x1:x2]
    #         imsave(img_path, face)
    #     except Exception as e:
    #         pass


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
