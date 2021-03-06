import os
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import model_from_json
from datetime import datetime
from keras.applications.vgg16 import decode_predictions
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks.tensorboard_v1 import TensorBoard
from skimage.exposure import rescale_intensity
import argparse
from keras import backend as K
import dlib
import model as m
from sklearn import model_selection

import visualize
from align import AlignDlib
import traceback
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import itertools
from keras.utils import to_categorical, np_utils
import matplotlib.pyplot as plt
from imageio import imread, imsave
import keras_vggface
import mtcnn
import numpy as np
import pandas as pd
from glob import glob
import scipy
from scipy import ndimage
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow_core.python.keras.optimizers import SGD
from tqdm import tqdm
import cv2
import random
import pickle
import time
from PIL import Image
import vgg_model

# IMPORTANT
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

DATA_DIR = r"D:\Data\train"
DATA_TEST_DIR = r"D:\Data\test"
DATAFRAME_PATH = r'dataframe_original.zip'
# alignment = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')
hog_detector = dlib.get_frontal_face_detector()


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


def save_pickle(array, name):
    pickle_out = open(f"{name}.pickle", "wb")
    pickle.dump(array, pickle_out)
    pickle_out.close()


def load_pickle(name):
    pickle_in = open(f"{name}", 'rb')
    return pickle_in.load(pickle_in)


def create_data_frame(data_path, save=False):
    df_train = pd.DataFrame(columns=['image', 'label', 'name'])
    train_path = glob(fr"{data_path}\*")
    print("\n", train_path)
    for index, train_path in tqdm(enumerate(train_path)):
        name = train_path.split('\\')[-1]
        print(f"\n{name}")
        images = glob(train_path + r"\*")
        if len(images) > 1:
            for image in images:
                df_train.loc[len(df_train)] = [image, index, name]
        else:
            continue
    print(len(df_train))
    if save is True:
        print("SAVING DATA-FRAME")
        compression_opts = dict(method='zip',
                                archive_name='dataframe.csv')
        df_train.to_csv(DATAFRAME_PATH, index=False, compression=compression_opts)

    return df_train


# mnist_1 = keras_vggface.vggface
def create_training_data(directory_path, data_name, label_name, required_size=(221, 221),
                         shuffle=False):
    if os.path.exists(DATAFRAME_PATH):
        df_train = pd.read_csv(DATAFRAME_PATH, compression='zip')
        print(f"\n{len(df_train)}")
    else:
        df_train = create_data_frame(directory_path, save=True)
        print(f"\n{len(df_train)}")
    training_data = []
    print(f"\nINFO={df_train.info()}")
    print(f"\nLENGTH={len(df_train)}")
    print(f"\nSHAPE={df_train.shape}")

    for index, row in tqdm(df_train.iterrows()):
        try:
            image = cv2.imread(row['image'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if type(image) is np.ndarray:
                if image.shape[:2] != required_size:
                    print(row['image'])
                    print(f"SHAPE DON'T MATCH={image.shape}")
                    img_array = cv2.resize(image, required_size, interpolation=cv2.INTER_LINEAR)
                    print(f"SHAPE RESIZED={image.shape}")
                training_data.append([image, row['label']])

        except Exception as e:
            print(f"ERROR-{row['image']}")
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            pass

    print(f"TRAINING DATA LENGTH: {len(training_data)}")
    num_classes = np.amax(np.array(training_data)[:, 1]) + 1
    print(f"CLASS NUMBER: {num_classes}")
    if shuffle is True:
        random.shuffle(training_data)

    x_trains = []
    y_trains = []

    for features, label in training_data:
        x_trains.append(features)
        y_trains.append(label)

    x_trains = np.array(x_trains)
    y_trains = np.array(y_trains)

    print(f"X:{x_trains.shape}")
    print(f"Y:{y_trains.shape}")

    np.save(f'{data_name}', x_trains)
    np.save(f'{label_name}', y_trains)


# Display one image
def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display two images
def display(a, b, title1="Original", title2="Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display 5 images
def display_5(original, images, df):
    count = 0
    plt.subplot(2, 3, 1), plt.imshow(original), plt.title(f"Original")
    rows = 2
    cols = 3
    images = list(reversed(images))

    for i in range(2, cols * rows + 1):
        if count == 5:
            break
        print(f"PERSON No.{images[count][0]} | LABEL No.{images[count][3]}")
        name = df[(df['label'] == images[count][3])].iloc[0, 2]
        plt.subplot(rows, cols, i), plt.imshow(images[count][2]),
        plt.title(f"Name:{name}\nEMB:{images[count][1]}")
        count = count + 1
    plt.tight_layout(pad=2.0)
    plt.show()


# def align_face(face, image_size):
#     (h, w, c) = face.shape
#     bb = dlib.rectangle(0, 0, w, h)
#     return alignment.align(image_size, face, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def create_model():
    x_train = np.load('x_test_221_shuffle.npy')
    y_train = np.load('y_test_221_shuffle.npy')

    print(f"X-TRAIN-SHAPE:{x_train.shape},\tDTYPE: {x_train.dtype}")
    print(f"Y-TRAIN-SHAPE:{y_train.shape},\tDTYPE: {y_train.dtype}")
    num_classes = np.amax(np.array(y_train)[:]) + 1
    print(f"\nCLASS NUMBER: {num_classes}")
    # classes = np.unique(y_train)

    vgg_model.batch_size = 9
    print(vgg_model.batch_size)
    model = vgg_model.deep_rank_model(input_shape=x_train.shape[1:])
    print("Loading pre-trained weight")
    weights_path = f'weights/triplet_weights_1_221_01.hdf5'
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("SUCCESSFULLY LOADED!!!")
        except ValueError as ve:
            print(ve)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            pass
    else:
        print(f"Paths don't exists, start training from scratch!")
    model.summary()
    model.compile(optimizer=tf.optimizers.SGD(lr=0.000001, decay=0.001, momentum=0.9, nesterov=True),
                  loss=vgg_model._loss_tensor)
    checkpoint = ModelCheckpoint("weights/triplet_weights_1_221_{epoch:02d}.hdf5",
                                 period=1,
                                 verbose=1,
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 mode='min')
    csv_logger = CSVLogger('csv_logger.log', separator=',', append=True)
    log_dir = fr".\logs\221\{str(datetime.now().strftime('%Y%m%d-%H%M%S'))}"
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, csv_logger, tensorboard]
    # tensorboard --logdir=./logs --host=127.0.0.1

    try:
        model.fit_generator(generator=vgg_model.image_batch_generator(x_train, y_train, vgg_model.batch_size),
                            steps_per_epoch=len(x_train) // vgg_model.batch_size,
                            epochs=400,
                            verbose=1,
                            callbacks=callbacks_list)
    except TypeError as te:
        print(te)
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        pass

    # model.save_weights("weights/best/vgg_best_weights.hdf5")


def predictor(image, embs, features, labels, df, model):
    global hog_detector
    images = []

    start = time.time()
    faces_hog = hog_detector(image, 1)
    end = time.time()
    print("Hog + SVM Execution time: " + str(end - start))

    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        frame = image[y:y + h, x:x + w]
        frame = cv2.resize(frame, (221, 221))
        frame = frame / 255.
        frame = np.expand_dims(frame, axis=0)

        emb = model.predict([frame, frame, frame])
        minimum = 10
        person = -1

        for i, e in enumerate(embs):
            dist = np.linalg.norm(emb - e)
            if dist < minimum:
                minimum = dist
                person = i
                images.append([i, minimum, features[i], labels[i]])
                print(f"{i} - {minimum}")

        emb = minimum
        name = df[(df['label'] == labels[person])].iloc[0, 2]
        print(f"\nPERSON: {person}  LABEL: {labels[person]} NAME: {name}")
        print(f"EMB: {emb}")

        cv2.putText(image, name, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, str(emb), (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        display_5(image, images, df)
        return images


# Preprocessing
def preprocessing(directory_path, save_path, required_size=(225, 225)):
    detector = dlib.get_frontal_face_detector()
    detecto_mtcnn = mtcnn.MTCNN()
    train_paths = glob(fr"{directory_path}\*")

    if not os.path.exists(f"{save_path}"):
        os.mkdir(fr"{save_path}")
    else:
        pass
    for path in tqdm(train_paths):
        name = path.split("\\")[-1]
        images = glob(f"{path}\\*")
        for image_path in images:
            try:
                temp_path = image_path.split("\\")[-1]
                image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_recs = detector(image, 1)
                face = None
                if len(face_recs) == 1:
                    x = face_recs[0].left()
                    y = face_recs[0].top()
                    w = face_recs[0].right() - x
                    h = face_recs[0].bottom() - y
                    face = cv2.resize(image[y:y + h, x:x + w], required_size, interpolation=cv2.INTER_LINEAR)

                else:
                    print(fr"{name}")
                    faces = detecto_mtcnn.detect_faces(image)
                    for face in faces:
                        x, y, width, height = face['box']
                        face = cv2.resize(image[y:y + height, x:x + width], required_size,
                                          interpolation=cv2.INTER_LINEAR)
                if not os.path.exists(fr"{save_path}\{name}"):
                    os.mkdir(fr"{save_path}\{name}")
                    cv2.imwrite(fr"{save_path}\{name}\{temp_path}", face)
                else:
                    cv2.imwrite(fr"{save_path}\{name}\{temp_path}", face)
            except Exception as e:

                print(e)
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                pass


def create_directory(p):
    # detector = dlib.get_frontal_face_detector()
    paths = glob(fr"{p}\*")
    for path in tqdm(paths):
        image = path.split("\\")[-1]
        name = image.split(".")[0]
        os.makedirs(fr"{p}\{name}")
        os.rename(fr"{path}", fr"{p}\{name}\{image}")


def evaluate(df_train, model, embs, labels):
    global hog_detector
    y_predict = []

    for index, row in tqdm(df_train.iterrows()):
        try:
            image = cv2.imread(row['image'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces_hog = hog_detector(image, 1)

            for face in faces_hog:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y

                frame = image[y:y + h, x:x + w]
                frame = cv2.resize(frame, (221, 221))
                frame = frame / 255.
                frame = np.expand_dims(frame, axis=0)

                emb = model.predict([frame, frame, frame])
                minimum = 10
                person = None

                for i, e in enumerate(embs):
                    dist = np.linalg.norm(emb - e)
                    if dist < minimum:
                        minimum = dist
                        person = i

                label = labels[person]
                print(f"\nPERSON:{person}")
                print(f"\nLABEL_PRED:{label}")
                print(f"\nLABEL_REAL:{row['label']}")

                if int(label) == int(row['label']):
                    print(f"\nPREDICT:True")
                    y_predict.append(True)
                else:
                    print(f"\nPREDICT:False")
                    y_predict.append(False)

        except cv2.error:
            pass
    np.save('y_pred', y_predict)


def calculate(y_pred):
    for i in y_pred:
        print(i)


def main():
    required_size = (221, 221)
    # create_directory(fr"D:\Data\2019")
    # preprocessing(r"D:\Data\2019", r"D:\Data\2019-faces-221", required_size=required_size)
    # create_data_frame(r'D:\Data\2019-faces-221', save=True)
    #
    # create_training_data(r'D:/Data/2019-faces', f'x_train_{required_size[0]}_2019',
    #                      f'y_train_{required_size[0]}_2019',
    #                      required_size=required_size, shuffle=True)
    # create_training_data(r'D:/Data/images/faces', f'x_test_{required_size[0]}_shuffle',
    #                      f'y_test_{required_size[0]}_shuffle',
    #                      required_size=required_size, shuffle=True)

    # create_model()
    # x_train = np.load('weights/x_test_221_shuffle.npy')
    # y_train = np.load('weights/y_test_221_shuffle.npy')

    # model = vgg_model.deep_rank_model(input_shape=(221, 221, 3))
    # model.load_weights(r"C:\FaceRecognition\weights\triplet_weights_5_221_58.hdf5")
    # model.summary()
    # model.save(r'C:\FaceRecognition\weights\triplet_mode_221.hdf5')

    # model = load_model("weights/triplet_models_3_221_35.h5", compile=False)
    # model.summary()
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.000001, decay=0.001, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # embs = []
    # for x in tqdm(x_train):
    #     image = x / 255.
    #     image = np.expand_dims(image, axis=0)
    #     emb = model.predict([image, image, image])
    #     embs.append(emb[0])
    #     del image
    # embs = np.array(embs)
    # print(embs.shape)
    # np.save('weights/embs335-test', embs)
    # embs = np.load('weights/embs335-test.npy')
    # df_train = pd.read_csv('weights/dataframe.zip')
    # print(len(df_train))
    # image = cv2.imread(r"D:/Data/res/pic/New folder/yeri.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # predictor(image, embs, x_train, y_train, df_train, model)

    # v = visualize.Visualize()
    # v.generate_sample('Yeri')
    # v.generate_sample('Irene')
    # v.generate_sample('Wendy')
    # v.generate_random_sample(1000)

    # create_data_frame(r'D:\Data\images\lfw', save=True)
    df_train = pd.read_csv('dataframe_original.zip')
    model = load_model("weights/triplet_models_4_221_60.h5", compile=False)
    model.compile(optimizer=tf.optimizers.SGD(lr=0.000001,
                                              decay=0.001,
                                              momentum=0.9,
                                              nesterov=True),
                  loss=vgg_model._loss_tensor)
    embs = np.load('weights/embs460-test.npy')
    y_test = np.load('weights/y_test_221_shuffle.npy')
    evaluate(df_train, model, embs, y_test)
    y_pred = np.load('y_pred.npy')
    # calculate(y_pred)


if __name__ == "__main__":
    main()
