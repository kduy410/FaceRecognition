import os

from keras.callbacks import ModelCheckpoint
from skimage.exposure import rescale_intensity
import argparse
from keras import backend as K
import dlib
import model as m
from sklearn import model_selection
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

DATA_DIR = r"C:\Data\temp"
DATA_TEST_DIR = r"C:\Data\temp"
alignment = AlignDlib('model/shape_predictor_68_face_landmarks.dat')


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


def create_data_frame(data_path):
    df_train = pd.DataFrame(columns=['image', 'label', 'name'])
    train_path = glob(fr"{data_path}\*")
    print("\n", train_path)
    for index, train_path in tqdm(enumerate(train_path)):
        name = train_path.split('\\')[-1]
        print(f"\n{name}")
        images = glob(train_path + r"\*")
        for image in images:
            df_train.loc[len(df_train)] = [image, index, name]
    return df_train


# mnist_1 = keras_vggface.vggface
def create_training_data(directory_path, data_name, label_name, required_size=(224, 224)):
    # CREATE TRAINING DATA FROM PREPROCESSED DATA
    df_train = create_data_frame(directory_path)
    training_data = []

    print(f"\n{df_train.head()}")
    print(f"\nINFO={df_train.info()}")
    print(f"\nLENGTH={len(df_train)}")
    print(f"\nSHAPE={df_train.shape}")

    for index, row in tqdm(df_train.iterrows()):
        try:
            img = Image.open(row['image'])
            img.load()
            img_array = np.array(img).astype(dtype=np.uint8)

            if type(img_array) is np.ndarray:
                if img_array.shape[-1] != 3:
                    if img_array.shape[-1] == 4:  # RGBA
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                        print(f"CONVERTED={img_array.shape}")
                    else:  # GRAYSCALE
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        print(f"CONVERTED={img_array.shape}")
                if img_array.shape[:2] != required_size:
                    print(f"SHAPE DON'T MATCH={img_array.shape}")
                    img_array = cv2.resize(img_array, required_size, interpolation=cv2.INTER_LINEAR)
                    print(f"SHAPE RESIZED={img_array.shape}")

                training_data.append([img_array, row['label']])

        except Exception as e:
            print(f"ERROR-{row['image']}")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            pass

    print(f"TRAINING DATA LENGTH: {len(training_data)}")
    num_classes = np.amax(np.array(training_data)[:, 1]) + 1
    print(f"CLASS NUMBER: {num_classes}")
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


def align_face(face, image_size):
    (h, w, c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    return alignment.align(image_size, face, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


# Preprocessing
def preprocessing(directory_path, save_path, required_size=(250, 250)):
    # create the detector, using default weights

    detector = mtcnn.MTCNN()
    train_paths = glob(fr"{directory_path}\*")

    for path in tqdm(train_paths):
        name = path.split("\\")[-1]
        images = glob(f"{path}\\*")
        for image_path in tqdm(images):
            try:
                temp_path = image_path.split("\\")[-1]
                # load image from file
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

                # ----------------------------------
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # ----------------------------------
                # Remove noise
                # Gaussian
                image = cv2.GaussianBlur(image, (5, 5), 0)

                # ----------------------------------
                # Denoising of image saving it into dst image
                image = cv2.addWeighted(image, 1.5, image, -0.5, 0, image)

                # ----------------------------------
                # Face Cropping
                # Detect faces in the image
                face_recs = detector.detect_faces(image)

                # Extract the bounding box from the first face
                x1, y1, width, height = face_recs[0]['box']
                x2, y2 = x1 + width, y1 + height

                # Extract the face
                face = image[y1:y2, x1:x2]

                # NORMALIZE
                # --------------------------------
                norm_face = cv2.normalize(face, required_size, 0, 255, cv2.NORM_MINMAX)

                # ----------------------------------
                # Face straightening
                face_rotated = align_face(norm_face, required_size[0])
                face_rotated = (face_rotated / 255.).astype(np.float32)

                # RESIZE
                # --------------------------------
                face_rotated = cv2.resize(face_rotated, required_size, interpolation=cv2.INTER_LINEAR)
                # Convert to array
                # --------------------------------
                face = np.asarray(face_rotated)
                face = tf.image.convert_image_dtype(face, dtype=tf.uint8, saturate=False)

                if not os.path.exists(fr"{save_path}/{name}"):
                    os.mkdir(fr"{save_path}/{name}")
                    imsave(fr"{save_path}/{name}/{temp_path}.jpeg", face, format='JPEG')
                else:
                    if os.path.exists(fr"{save_path}/{name}/{temp_path}.jpeg"):
                        imsave(fr"{save_path}/{name}/{temp_path}-{int(time.time())}.jpeg", face, format='JPEG')
                    else:
                        imsave(fr"{save_path}/{name}/{temp_path}.jpeg", face, format='JPEG')
            except Exception as e:
                print(e)
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                pass


def seq_model(input_shape=(224, 224, 3), classes=10):
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(64, input_shape=input_shape, activation='relu'),
        Dense(128, activation='relu'),
        Dense(classes, activation='softmax')
    ])


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Prediction label")


def _loss_tensor(y_true, y_pred):
    _EPSILON = K.epsilon()
    batch_size = 32
    y_pred = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)
    loss = 0.
    g = 1.
    for i in range(0, batch_size, 3):
        try:
            q_embedding = y_pred[i]
            p_embedding = y_pred[i + 1]
            n_embedding = y_pred[i + 2]
            D_q_p = K.sqrt(K.sum((q_embedding - p_embedding) ** 2))
            D_q_n = K.sqrt(K.sum((q_embedding - n_embedding) ** 2))
            loss = loss + g + D_q_p - D_q_n
        except:
            continue
    loss = loss / batch_size * 3
    return K.maximum(loss, 0)


def create_model():
    # NAME = f"People-cnn-64x2-{int(time.time())}"
    # tensorboard = TensorBoard(log_dir=f'logs\{NAME}')
    x_train = np.load('x_train_128_lfw.npy')
    y_train = np.load('y_train_128_lfw.npy')
    x_train = x_train / 255.0
    y_train = y_train / 255.0
    print(f"X-TRAIN-SHAPE:{x_train.shape},\tDTYPE: {x_train.dtype}")
    print(f"Y-TRAIN-SHAPE:{y_train.shape},\tDTYPE: {y_train.dtype}")
    # num_classes = int(np.amax(np.array(y_train)[:]) + 1)
    # print(f"\nCLASS NUMBER: {num_classes}")
    # classes = np.unique(y_train)

    batch_size = 24
    model = vgg_model.deep_rank_model(input_shape=x_train.shape[1:])
    print("Loading pre-trained weight")
    weights_path = 'weights/triplet_weights.hdf5'
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print('False')
        return

    model.summary()
    model.compile(optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
                  loss=vgg_model._loss_tensor)

    checkpoint = ModelCheckpoint("weights/triplet_weights-{epoch:02d}.hdf5",
                                 period=1,
                                 monitor='loss',
                                 verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]
    model.fit_generator(generator=vgg_model.image_batch_generator(x_train, y_train, batch_size),
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=40,
                        verbose=1,
                        callbacks=callbacks_list)

    # evaluate the model
    # scores = model.evaluate([x_train, x_train, x_train], y_train, verbose=1)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # model.save('model/triplet_model.h5')
    # model.save_weights("vgg_model_weights.h5")


def main():
    # required_size = (128, 128)
    # # preprocessing(r"C:\Data\lfw_funneled", r"C:\Data\temp", required_size=required_size)
    # # preprocessing(r"C:\Data\lfw-deepfunneled", r"C:\Data\temp", required_size=required_size)
    # # create_training_data(DATA_DIR, f'x_train_{required_size[0]}_lfw', f'y_train_{required_size[0]}_lfw',
    # #                      required_size=required_size)
    # # create_training_data(DATA_TEST_DIR, f'x_test_{required_size[0]}_lfw', f'y_test_{required_size[0]}_lfw',
    # #                      required_size=required_size)
    # # create_model()
    # # df = create_data_frame(r'C:\Data\temp')
    # # df.to_csv('out.csv')
    # df = pd.read_csv("out.csv")
    # model = vgg_model.deep_rank_model(input_shape=(128, 128, 3))
    # model.load_weights('weights/triplet_weights-01.hdf5')
    # model.summary()
    # image = cv2.imread(r'C:\Data\Pictures\c.jpg', cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(np.array(image).shape)
    # detector = mtcnn.MTCNN()
    # # Detect faces in the image
    # face_recs = detector.detect_faces(image)
    #
    # # Extract the bounding box from the first face
    # x1, y1, width, height = face_recs[0]['box']
    # x2, y2 = x1 + width, y1 + height
    #
    # # Extract the face
    # face = image[y1:y2, x1:x2]
    # print(np.array(face).shape)
    # frame = face
    # frame = cv2.resize(frame, (128, 128))
    # frame = frame / 255.
    # frame = np.expand_dims(frame, axis=0)
    # x_test = np.load('x_test_128.npy')
    # emb128 = model.predict([x_test[:1], x_test[:1], x_test[:1]])
    # print(emb128)
    # print(np.array(frame).shape)
    # minimum = 99999
    # person = -1
    # embs128 = np.load('x_train_128_lfw.npy')
    # labels = np.load('y_train_128_lfw.npy')
    # for k, e in tqdm(enumerate(embs128)):
    #     # Euler distance
    #     dist = np.linalg.norm(emb128.shape[1:] - e)
    #     if dist < minimum:
    #         minimum = dist
    #         person = k
    # id = labels[person]
    # print(id)
    # name = df[(df['label'] == labels[person])].iloc[0, 3]
    # print(name)
    # detected = cv2.cvtColor(cv2.imread(df[(df['label'] == labels[person])].iloc[0, 1]
    #                                    , cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    # cv2.putText(image, name, (x1 - 10, y1 - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # display(image, detected)
    x = np.load('x_train_128_lfw.npy')
    y = np.load('y_test_128_lfw.npy')

    for _ in [1]:
        batch_paths = np.random.choice(a=len(x), size=24 // 3)
        print(f'BATCH-{batch_paths}')
        input_1 = []
        for i in batch_paths:
            print('I:', i)
            pos = np.where(y == y[i])[0]
            print(f"POS:{pos}")
            print(f"POS-LEN:{len(pos)}")
            neg = np.where(y != y[i])[0]
            print(f"NEG:{neg}")
            print(f"NEG-LEN:{len(neg)}")
            if len(pos) is not 1:
                j = np.random.choice(pos)
                print(f"J:{j}")
                while j == i:
                    print(f"J==I:{j}")
                    j = np.random.choice(pos)
            else:
                j = i
                print(f"J!=I:{j}")

            k = np.random.choice(neg)
            print(f"NEG-K:{k}")

            while k == i:
                print(f"K==I:{k}")
                k = np.random.choice(neg)

            input_1.append(y[i])
            input_1.append(y[j])
            input_1.append(y[k])

        input_1 = np.array(input_1)
        input = [input_1, input_1, input_1]
        print(input)


if __name__ == "__main__":
    main()
