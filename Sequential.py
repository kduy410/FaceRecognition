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
# check version of keras_vggface
# print version
print("keras_vggface-version :", keras_vggface.__version__)
# confirm mtcnn was installed correctly
# print version
print("MTCNN-version :", mtcnn.__version__)

DATA_DIR = r"D:\Data\train"
DATA_TEST_DIR = r"D:\Data\test"
DATAFRAME_PATH = r'dataframe.zip'
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


# def align_face(face, image_size):
#     (h, w, c) = face.shape
#     bb = dlib.rectangle(0, 0, w, h)
#     return alignment.align(image_size, face, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


# Preprocessing
def preprocessing(directory_path, save_path, required_size=(221, 221)):
    detector = dlib.get_frontal_face_detector()
    train_paths = glob(fr"{directory_path}\*")

    for path in tqdm(train_paths):
        name = path.split("\\")[-1]
        images = glob(f"{path}\\*")
        for image_path in tqdm(images):
            try:
                temp_path = image_path.split("\\")[-1]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_recs = detector(image, 1)
                face = None
                if len(face_recs) == 1:
                    x = face_recs[0].left()
                    y = face_recs[0].top()
                    w = face_recs[0].right() - x
                    h = face_recs[0].bottom() - y
                    face = cv2.resize(image[y:y + h, x:x + w], required_size, interpolation=cv2.INTER_LINEAR)
                else:
                    continue
                if not os.path.exists(fr"{save_path}/{name}"):
                    os.mkdir(fr"{save_path}/{name}")
                    imsave(fr"{save_path}/{name}/{temp_path}", face)
                else:
                    imsave(fr"{save_path}/{name}/{temp_path}", face)
            except Exception as e:
                print(e)
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                pass


def create_model():
    x_train = np.load('x_train_221_shuffle.npy')
    y_train = np.load('y_train_221_shuffle.npy')

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
    model.compile(optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
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


def predictor(image, embs, labels, df, model):
    global hog_detector

    start = time.time()
    faces_hog = hog_detector(image, 1)
    end = time.time()
    print("Hog + SVM Execution time: " + str(end - start))

    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        # draw green box over face which detect by hog + svm
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get vector embeding
        frame = image[y:y + h, x:x + w]
        frame = cv2.resize(frame, (96, 96))
        # frame = align_face(frame, 96)
        # frame = (frame / 255.).astype(np.float32)
        # frame = frame / 255.
        display_one(frame)
        frame = np.expand_dims(frame, axis=0)
        emb = model.predict([frame, frame, frame])

        minimum = 99999

        person = -1

        for i, e in enumerate(embs):
            # Euler distance
            dist = np.linalg.norm(emb - e)
            if dist < minimum:
                minimum = dist
                person = i
                print(i)
                print(minimum)
        EMB = minimum
        print("\nPERSON: ", person)
        print("\nPERSON-LABEL: ", labels[person])
        name = df[(df['label'] == labels[person])].iloc[0, 2]
        print("\nPERSON-NAME: ", name)
        print("\nPERSON-EMB: ", EMB)
        # convert the probabilities to class labels
        # print('%s (%.2f%%)' % (name, EMB))
        cv2.putText(image, name, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, str(EMB), (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        display_one(image)


def main():
    required_size = (221, 221)

    # preprocessing(r"D:\Data\images\lfw", r"D:\Data\images\faces", required_size=required_size)
    create_data_frame('D:/Data/images/faces', save=True)
    # create_training_data(r'D:/Data/images/faces', f'x_train_{required_size[0]}_shuffle',
    #                      f'y_train_{required_size[0]}_shuffle',
    #                      required_size=required_size, shuffle=True)
    create_training_data(r'D:/Data/images/faces', f'x_test_{required_size[0]}_shuffle',
                         f'y_test_{required_size[0]}_shuffle',
                         required_size=required_size, shuffle=True)

    # create_model()
    # x_train = np.load('x_train_96_1_shuffle.npy')
    # y_train = np.load('y_train_96_1_shuffle.npy')
    #
    # model = vgg_model.deep_rank_model(input_shape=x_train.shape[1:])
    # model.load_weights(r"C:\FaceRecognition\weights\models-DENSEFINAL96\triplet_weights-192-8-96.hdf5")
    # model.summary()

    # embs96 = []
    # for x in tqdm(x_train):
    #     # image = x / 255.
    #     image = np.expand_dims(x, axis=0)
    #     emb128 = model.predict([image, image, image])
    #     embs96.append(emb128[0])
    #     del image
    # embs96 = np.array(embs96)
    # print(embs96.shape)
    # np.save('embs96', embs96)
    # embs = np.load('embs96.npy')
    # df_train = pd.read_csv('dataframe_1.zip')
    # print(len(df_train))
    #
    # image = cv2.imread(r'D:\Data\aaron.jpg', cv2.IMREAD_UNCHANGED)
    # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # imsave(r"D:\Data\temp", format="JPEG")
    # img = Image.open(r"D:\Data\temp")
    # img.load()
    # img_array = np.array(img).astype(dtype=np.uint8)
    #
    # predictor(image, embs, y_train, df_train, model)

    # v = visualize.Visualize()
    # v.generate_sample('Yeri')
    # v.generate_sample('Irene')
    # v.generate_sample(5000)


if __name__ == "__main__":
    main()
