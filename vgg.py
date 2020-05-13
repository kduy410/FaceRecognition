import os
from datetime import datetime
from scipy.misc import face
from scipy.ndimage import rotate, zoom
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
from keras.layers import Input, Lambda
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
# DATAFRAME_PATH = r'dataframe_original.zip'
# alignment = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')
hog_detector = dlib.get_frontal_face_detector()
detector_mtcnn = mtcnn.MTCNN()


def create_data_frame(data_path, save_path, save=False):
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
    if save == True:
        print("SAVING DATA-FRAME")
        compression_opts = dict(method='zip',
                                archive_name='dataframe.csv')
        df_train.to_csv(save_path, index=False, compression=compression_opts)

    return df_train


def create_training_data(directory_path, DATAFRAME_PATH, data_name, label_name, required_size=(220, 220),
                         shuffle=False):
    if os.path.exists(DATAFRAME_PATH):
        df_train = pd.read_csv(DATAFRAME_PATH, compression='zip')
        print(f"\n{len(df_train)}")
    else:
        return
    training_data = []

    print(f"\nINFO={df_train.info()}")
    print(f"\nLENGTH={len(df_train)}")
    print(f"\nSHAPE={df_train.shape}")

    for index, row in tqdm(df_train.iterrows()):
        try:
            image = cv2.imread(row['image'])
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # display_one(image)
            if type(image) == np.ndarray:
                if image.shape[:2] != required_size:
                    print(row['image'])
                    print(f"SHAPE DON'T MATCH={image.shape}")
                    image = cv2.resize(image, required_size, interpolation=cv2.INTER_LINEAR)
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

    if shuffle == True:
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
def preprocessing(directory_path, save_path, original_save_path, required_size=(220, 220)):
    detector = dlib.get_frontal_face_detector()
    train_paths = glob(fr"{directory_path}\*")

    if not os.path.exists(f"{save_path}"):
        os.mkdir(fr"{save_path}")
    else:
        pass

    if not os.path.exists(f"{original_save_path}"):
        os.mkdir(fr"{original_save_path}")
    else:
        pass

    for path in tqdm(train_paths):
        name = path.split("\\")[-1]
        images = glob(f"{path}\\*")
        for image_path in images:
            try:
                temp_path = image_path.split("\\")[-1]
                image = cv2.imread(image_path)
                original = image
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_recs = detector(image, 1)
                if len(face_recs) == 1:
                    x = face_recs[0].left()
                    y = face_recs[0].top()
                    w = face_recs[0].right() - x
                    h = face_recs[0].bottom() - y
                    face = cv2.resize(image[y:y + h, x:x + w], required_size, interpolation=cv2.INTER_LINEAR)
                    if not os.path.exists(fr"{save_path}/{name}"):
                        os.mkdir(fr"{save_path}/{name}")
                        cv2.imwrite(fr"{save_path}/{name}/{temp_path}", face)
                    else:
                        cv2.imwrite(fr"{save_path}/{name}/{temp_path}", face)

                    if not os.path.exists(fr"{original_save_path}/{name}"):
                        os.mkdir(fr"{original_save_path}/{name}")
                        cv2.imwrite(fr"{original_save_path}/{name}/{temp_path}", original)
                    else:
                        cv2.imwrite(fr"{original_save_path}/{name}/{temp_path}", original)
                else:
                    continue
            except Exception as e:
                print(e)
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                pass


def create_model():
    x_train = np.load('x_train_220_BGR.npy')
    y_train = np.load('y_train_220_BGR.npy')

    print(f"X-TRAIN-SHAPE:{x_train.shape},\tDTYPE: {x_train.dtype}")
    print(f"Y-TRAIN-SHAPE:{y_train.shape},\tDTYPE: {y_train.dtype}")

    num_classes = np.amax(np.array(y_train)[:]) + 1
    print(f"\nCLASS NUMBER: {num_classes}")
    # classes = np.unique(y_train)

    vgg_model.batch_size = 9

    # model = load_model(f'weights/triplet_models_BGR.h5',
    #                    compile=False)

    model = vgg_model.deep_rank_model(input_shape=x_train.shape[1:])
    print("Loading pre-trained weight")
    weights_path = f'weights/triplet_weights_BGR.h5'

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
    SGD = tf.optimizers.SGD(lr=0.1, decay=1e-06, momentum=0.9, nesterov=True)
    model.compile(optimizer=SGD, loss=vgg_model._loss_tensor)

    checkpoint = ModelCheckpoint("weights/triplet_weights_BGR.h5",
                                 period=1,
                                 verbose=1,
                                 monitor='loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 mode='min')

    csv_logger = CSVLogger('csv_logger.log', separator=',', append=True)
    log_dir = fr".\logs\220-bgr\{str(datetime.now().strftime('%Y%m%d-%H%M%S'))}"
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [checkpoint, csv_logger, tensorboard]
    # tensorboard --logdir=./logs --host=127.0.0.1

    try:
        model.fit_generator(generator=vgg_model.image_batch_generator(x_train, y_train,
                                                                      vgg_model.batch_size),
                            steps_per_epoch=len(x_train) // vgg_model.batch_size,
                            epochs=101,
                            verbose=1,
                            shuffle=True,
                            callbacks=callbacks_list)
    except TypeError as te:
        print(te)
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        pass


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

        frame = image[y:y + h, x:x + w]
        frame = cv2.resize(frame, (221, 221))
        frame = frame / 255.
        # display_one(frame)
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
        cv2.putText(image, str(name), (x, y + w + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # cv2.putText(image, str(EMB), (x, y + w + 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    display_one(image)


def evaluate(df_train, model, embs, labels):
    global hog_detector, detector_mtcnn
    y_predict = []
    x_train = np.load('x_train_221_BGR.npy')

    for index, row in tqdm(df_train.iterrows()):

        try:
            image = cv2.imread(row['image'])
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces_hog = hog_detector(image, 1)
            if len(faces_hog) == 1:
                for face in faces_hog:
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y

                    frame = image[y:y + h, x:x + w]
                    frame = cv2.resize(frame, (221, 221), interpolation=cv2.INTER_LINEAR)
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
                    print(person)
                    # print(f"\n{row['label']}")
                    label = labels[person]
                    print(f"\nLABEL:{label}")
                    # display_one(x_train[person])
                    name = df_train[(df_train['label'] == label)].iloc[0, 2]
                    print(f"\nPERSON:{person}")
                    print(f"\nNAME:{str(name)}")
                    print(f"\nLABEL_PRED:{label}")
                    print(f"\nLABEL_REAL:{row['label']}")

                    if int(label) == int(row['label']):
                        print(f"\nPREDICT:True")
                        y_predict.append(True)
                    else:
                        print(f"\nPREDICT:False")
                        y_predict.append(False)
            elif len(faces_hog) == 0:
                faces = detector_mtcnn.detect_faces(image)
                if len(faces) == 0:
                    continue
                else:
                    for face in faces:
                        x, y, width, height = face['box']

                        frame = image[y:y + height, x:x + width]
                        frame = cv2.resize(frame, (221, 221), interpolation=cv2.INTER_LINEAR)
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
                        name = df_train[(df_train['label'] == label)].iloc[0, 2]
                        print(f"\nPERSON:{person}")
                        print(f"\nNAME:{str(name)}")
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
    count = 0
    for i in y_pred:
        if i == True:
            count += 1
    print(count)
    print(f"PERCENT-ACCURACY:{round(float((count * 100) / len(y_pred)), 2)}")


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def create_directory(p):
    paths = glob(fr"{p}\*")
    for path in tqdm(paths):
        image = path.split("\\")[-1]
        name = image.split(".")[0]
        os.makedirs(fr"{p}\{name}")
        os.rename(fr"{path}", fr"{p}\{name}\{image}")


def generate_data(data_path):
    zoom_list = [0.75, 1, 1.25, 1.5]
    paths = glob(fr"{data_path}\*")

    for index, path in tqdm(enumerate(paths)):
        name = path.split("\\")[-1]
        images = glob(path + r"\*")
        for image_path in images:
            image = cv2.imread(image_path)
            new_name = name.split(".")[0]
            for i in range(0, 2):
                rot = rotate(image, random.randrange(-15, 15), reshape=False)
                zoom = cv2_clipped_zoom(rot, random.choice(zoom_list))
                # print(f"{path}\{new_name}_{str(i)}.jpg")
                cv2.imwrite(f"{path}\{new_name}_{str(i)}.jpg", zoom)


def main():
    required_size = (220, 220)
    # create_directory(r"D:\Data\images\2019")
    # generate_data(r"D:\Data\images\2019")
    # preprocessing(r"D:\Data\images\2019", r"D:\Data\images\2019-faces", r"D:\Data\images\2019-original",
    #               required_size=(220, 220))

    # create_data_frame(r'D:\Data\images\2019-faces', '2019-faces.zip', save=True)
    # create_data_frame(r'D:\Data\images\2019-original', '2019-original.zip', save=True)

    # create_training_data(r'D:\Data\images\2019-faces', '2019-faces.zip', f'x_train_{220}_BGR',
    #                      f'y_train_{220}_BGR',
    #                      required_size=(220, 220), shuffle=False)

    create_model()

    # x_train = np.load('x_train_220_BGR.npy')
    # y_train = np.load('y_train_220_BGR.npy')

    # model = vgg_model.deep_rank_model(input_shape=(221, 221, 3))
    # model.load_weights(r"C:\FaceRecognition\weights\triplet_weights_5_221_58.hdf5")
    # model.summary()
    # model.save(r'C:\FaceRecognition\weights\triplet_mode_221.hdf5')
    # model = load_model("weights/triplet_models_BGR.h5", compile=False)
    # model.summary()
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.01, decay=0.01, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.000001, decay=0.001, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.01, decay=0.001, momentum=0.9, nesterov=True),
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
    # np.save('embsBGR', embs)

    # embs = np.load('embsBGR.npy')
    # df_train = pd.read_csv('dataframe_faces_250.zip')
    # print(len(df_train))
    # image = cv2.imread(r"C:\Data\irene.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # predictor(image, embs, y_train, df_train, model)

    # v = visualize.Visualize()
    # v.generate_sample('Yeri')
    # v.generate_sample('Irene')
    # v.generate_sample('Wendy')
    # v.generate_random_sample(1000)

    # create_directory(fr"D:\Data\2019")
    # create_data_frame(r'D:\Data\images\faces', save=True)

    # df_train = pd.read_csv('dataframe_original.zip')
    # model = load_model("weights/triplet_models_BGR.h5", compile=False)
    # model.compile(optimizer=tf.optimizers.SGD(lr=0.01, decay=0.01, momentum=0.9, nesterov=True),
    #               loss=vgg_model._loss_tensor)
    # embs = np.load('embsBGR.npy')
    # y_test = np.load('y_train_221_BGR.npy')
    # evaluate(df_train, model, embs, y_test)
    # y_pred = np.load('y_pred.npy')
    # calculate(y_pred)


if __name__ == "__main__":
    main()
