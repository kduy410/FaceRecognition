import os

import dlib
import traceback
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import itertools
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from imageio import imread, imsave
import keras_vggface
import mtcnn
import numpy as np
import pandas as pd
from glob import glob
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
from tensorflow_core.python.keras.optimizers import SGD
from tqdm import tqdm
import cv2
import random
import pickle
import time
from PIL import Image

# check version of keras_vggface
# print version
print("keras_vggface-version :", keras_vggface.__version__)
# confirm mtcnn was installed correctly
# print version
print("MTCNN-version :", mtcnn.__version__)

DATA_DIR = r"D:\Data\train"
DATA_TEST_DIR = r"D:\Data\test"


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
def create_training_data_sequential(DATA_DIR, IMG_SIZE, data_name, label_name):
    df_train = create_data_frame(DATA_DIR)

    print(df_train.head())

    training_data = []

    print(f"\nINFO={df_train.info()}")
    print(f"\nLENGTH={len(df_train)}")
    print(f"\nSHAPE={df_train.shape}")

    for index, row in tqdm(df_train.iterrows()):
        try:
            img_array = imread(row['image'])
            if img_array.ndim is 2:
                img_gray = cv2.imread(row['image'], cv2.IMREAD_GRAYSCALE)
                img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                resized = prewhiten(cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
            else:
                resized = prewhiten(resize(img_array, (IMG_SIZE, IMG_SIZE)))
            # TEMPORARY USE GRAY_SCALE, COLOR DON'T WORK
            resized = skimage.color.rgb2gray(resized)
            training_data.append([resized, row['label']])
        except Exception as e:
            print(f"ERROR-{row['image']}")
            print(e)
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            pass

    print(f"TRAINING DATA LENGTH: {len(training_data)}")
    num_classes = np.amax(np.array(training_data)[:, 1])
    print(f"CLASS NUMBER: {num_classes}")
    random.shuffle(training_data)

    x_trains = []
    y_trains = []

    for features, label in training_data:
        x_trains.append(features)
        y_trains.append(label)

    # 1 for gray-scale
    # 3 for color
    # convert to array only work with gray_scale
    x_trains = np.array(x_trains)
    print(x_trains.shape)
    x_trains = x_trains.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print(x_trains.shape)
    y_trains = np.array(y_trains)

    print(f"X:{x_trains.shape}")
    print(f"Y:{y_trains.shape}")

    np.save(f'{data_name}', x_trains)
    np.save(f'{label_name}', y_trains)


def extracting_face_from_image(required_size=(224, 224)):
    # create the detector, using default weights
    detector = mtcnn.MTCNN()
    # hog = dlib.get_frontal_face_detector()
    # cnn = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')
    train_paths = glob(fr"{DATA_DIR}\*")

    for path in tqdm(train_paths):
        name = path.split("\\")[-1]
        images = glob(f"{path}\\*")
        for image_path in tqdm(images):
            try:
                # print(image_path)
                # load image from file
                image = imread(image_path)
                temp_path = image_path.split("\\")[-1]

                # detect faces in the image
                face_recs = detector.detect_faces(image)
                # face_recs = hog(image, 0)
                # face_rec = cnn(image, 1)

                # extract the bounding box from the first face
                x1, y1, width, height = face_recs[0]['box']
                x2, y2 = x1 + width, y1 + height

                # for hog or cnn, if it is cnn the face_rec.rect.left()...
                face_rec = face_recs[0]['box']
                # face_rec = face_recs[0]

                # if face_rec is None:
                #     continue
                #
                # x1 = face_rec.left()
                # y1 = face_rec.top()
                # x2 = face_rec.right()
                # y2 = face_rec.bottom()

                # extract the face
                face = image[y1:y2, x1:x2]
                # resize pixels to the model size
                image = Image.fromarray(face)
                image = image.resize(required_size)
                face = np.asarray(image)

                if not os.path.exists(fr"D:/Data/temp/{name}"):
                    os.mkdir(fr"D:/Data/temp/{name}")
                imsave(fr"D:/Data/temp/{name}/{temp_path}", face, format='JPG')
            except Exception as e:
                print(e)
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                pass


def new_sequential_model():
    NAME = f"People-cnn-64x2-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f'logs\{NAME}')

    x_trains = np.load('x_test.npy')
    y_trains = np.load('y_test.npy')

    print(f"X-SHAPE:{x_trains.shape},\nX-DTYPE: {x_trains.dtype}")
    print(f"Y-SHAPE:{y_trains.shape},\nY-DTYPE: {y_trains.dtype}")

    num_classes = np.amax(np.array(y_trains)[:]) + 1
    print(f"CLASS NUMBER: {num_classes}")

    # x_trains = l2_normalize(x_trains)

    model = Sequential([
        Flatten(),
        Dense(64, input_shape=x_trains.shape[1:], activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_trains, y_trains, batch_size=32, validation_split=0.2, epochs=15, shuffle=True, verbose=1,
              callbacks=[tensorboard])
    model.summary()
    # evaluate the model
    scores = model.evaluate(x_trains, y_trains, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # serialize weights to HDF5
    # model.save_weights("sequential.h5")
    # print("Saved model to disk")
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    predictions = model.predict_classes(x_test, batch_size=16, verbose=1)
    for i in predictions:
        print(i)
    cm = confusion_matrix(y_test, predictions)
    cm_plot_labels = ['no side effects', 'had side effects']
    plot_confusion_matrix(cm,cm_plot_labels,predictions)


def sequential_model():
    mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Scale
    # Easy for network to learn
    # Reduce time
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    # Build model
    model = tf.keras.models.Sequential()
    # Input layer
    model.add(Flatten(input_shape=x_train.shape[1:]))
    # Hidden layer: 128 neurons or units
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))
    # Output layer: 10 classification
    model.add(Dense(10, activation=tf.nn.softmax))
    # Parameter
    # Loss metrics is the decay of error - is what you got wrong
    model.compile(optimizer='adam',
                  loss=tf.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # To train the model -> model.fit
    model.fit(x_train, y_train, epochs=3)
    # Should expect out_of_sample accuracy to be slightly lower and loss to be slightly higher
    # Calculate loss, accuracy
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f"Val loss: {val_loss} - Val accuracy: {val_acc}")
    # print(plt.imshow(x_train[0]))

    # Save model
    model.save('num_reader_model.model')
    # Create new model
    new_model = tf.keras.models.load_model('num_reader_model.model')
    # Prediction  - predict always take a list
    predictions = new_model.predict([x_test])
    print(np.argmax(predictions[70]))
    plt.imshow(x_test[70])
    plt.show()


def main():
    # extracting_face_from_image()
    # create_training_data_sequential(DATA_DIR, 224, 'x_train', 'y_train')
    # create_training_data_sequential(DATA_TEST_DIR, 224, 'x_test', 'y_test')

    new_sequential_model()

    # model = t('sequential.h5')
    # # Prediction  - predict always take a list
    # test = imread("D:/Data/test/irene.jpg")
    # test = skimage.color.rgb2gray(test)
    # x = cv2.resize(test, (224, 224))
    # x = np.array(x).reshape(-1, 224, 224, 1)
    # print(f"{x.shape}")
    #
    # if len(x.shape) == 3:
    #     plt.imshow(np.squeeze(x), cmap='gray')
    # elif len(x.shape) == 2:
    #     plt.imshow(x, cmap='gray')
    # elif len(x.shape) == 4:
    #     plt.imshow(np.squeeze(x), cmap='gray')
    # else:
    #     print("Higher dimensional data")
    # plt.show()
    # predictions = model.predict(x)
    # print(f"PREDICTION :{np.argmax(predictions[0])}")


if __name__ == "__main__":
    main()
