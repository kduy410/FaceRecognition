import os

from skimage.exposure import rescale_intensity
import argparse
import dlib
from align import AlignDlib
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

# check version of keras_vggface
# print version
print("keras_vggface-version :", keras_vggface.__version__)
# confirm mtcnn was installed correctly
# print version
print("MTCNN-version :", mtcnn.__version__)

DATA_DIR = r"D:\Data\train"
DATA_TEST_DIR = r"D:\Data\test"
alignment = AlignDlib('weights/shape_predictor_68_face_landmarks.dat')


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
                    img_array = cv2.addWeighted(img_array, 1.5, img_array, -0.5, 0, img_array)
                    img_array = cv2.normalize(img_array, required_size, 0, 255, cv2.NORM_MINMAX)
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
    return alignment.align(image_size, face, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)


# Preprocessing
def preprocessing(directory_path, save_path, required_size=(224, 224)):
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
                    imsave(fr"{save_path}/{name}/{temp_path}", face, format='JPEG')
                else:
                    imsave(fr"{save_path}/{name}/{temp_path}", face, format='JPEG')
            except Exception as e:
                print(e)
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                pass


def new_sequential_model():
    NAME = f"People-cnn-64x2-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f'logs\{NAME}')

    x_trains = np.load('x_train.npy')
    y_trains = np.load('y_train.npy')

    print(f"X-SHAPE:{x_trains.shape},\nX-DTYPE: {x_trains.dtype}")
    print(f"Y-SHAPE:{y_trains.shape},\nY-DTYPE: {y_trains.dtype}")

    num_classes = np.amax(np.array(y_trains)[:]) + 1
    print(f"CLASS NUMBER: {num_classes}")

    # x_trains = l2_normalize(x_trains)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
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

    model.save('model.h5')


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
    # preprocessing(r"D:\Data\test", r"D:\Data\test")
    # create_training_data_sequential(DATA_DIR, 224, 'x_train', 'y_train')
    # create_training_data(DATA_DIR, 'x_train', 'y_train')
    create_training_data(DATA_TEST_DIR, 'x_test', 'y_test')

    # new_sequential_model()
    # model = load_model('model.h5')
    # model.summary()
    # model.get_weights()
    # model.optimizer
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
