import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import keras_vggface
import mtcnn
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import cv2
import random
import pickle
import time

# check version of keras_vggface
# print version
print("keras_vggface-version :", keras_vggface.__version__)
# confirm mtcnn was installed correctly
# print version
print("MTCNN-version :", mtcnn.__version__)


# mnist_1 = keras_vggface.vggface
def create_training_data_sequential():
    DATA_DIR = "D:/Data/train"
    df_train = pd.DataFrame(columns=['image', 'label', 'name'])
    train_path = glob(r"D:\Data\train\*")
    print("\n", train_path)
    for index, train_path in tqdm(enumerate(train_path)):
        name = train_path.split('\\')[-1]
        print(f"\n{name}")
        images = glob(train_path + r"\*")
        # print("\n", images)
        for image in images:
            # print("\n Length :", len(df_train))
            df_train.loc[len(df_train)] = [image, index, name]
    # print("\n Length :", len(df_train))
    print(df_train.head())

    # for img in df_train.loc[:, 'image']:
    #     img_array = cv2.imread(img, cv2.IMREAD_COLOR)
    #     # Extract color to change from RGB to BGR
    #     b, g, r = cv2.split(img_array)
    #     img_array = cv2.merge([r, g, b])
    #     print(img_array.shape)
    #     plt.imshow(img_array)
    #     plt.show()
    #     IMG_SIZE = 70
    #     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #     plt.imshow(new_array)
    #     plt.show()
    #     break
    IMG_SIZE = 90
    training_data = []
    print(f"\nINFO={df_train.info()}")
    print(f"\nLENGTH={len(df_train)}")
    print(f"\nSHAPE={df_train.shape}")
    # print(f"\nDF={df_train[['label', 'name']]}")
    for index, row in tqdm(df_train.iterrows()):
        # print(f"{row['image']}")
        # print(f"{index}")
        try:
            # GRAY-SCALE
            img_array = cv2.imread(row["image"], cv2.IMREAD_GRAYSCALE)
            # b, g, r = cv2.split(img_array)
            # img_array = cv2.merge([r, g, b])
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            # print(f"{isinstance(row['label'], list)}")

            training_data.append([new_array, row['label']])
        except Exception as e:
            print(f"{e}")
            pass
    # for sample in training_data[:10]:
    #     print(sample[1])
    print(f"TRAINING DATA LENGTH: {len(training_data)}")
    random.shuffle(training_data)
    x = []
    y = []

    for features, label in training_data:
        x.append(features)
        y.append(label)

    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    print(f"X:{x.shape}")
    print(f"Y:{y.shape}")

    pickle_out = open("x.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    pickle_in = open("x.pickle", 'rb')
    x = pickle.load(pickle_in)


def new_sequential_model():
    NAME = f"People-cnn-64x2-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f'logs\{NAME}')
    x = pickle.load(open("x.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)

    x = x / 255.0
    # y = y / 255.0

    model = Sequential()
    # CONV2D 64 unit, windows size 3.3
    model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2 by 64 layer, final layer is dense, must flatten the data to 1D
    model.add(Flatten())
    model.add(Dense(64))
    # Output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    model.fit(x, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])

    # Save model

    model.save('sequential.model')


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
    # create_training_data_sequential()
    # new_sequential_model()
    # sequential_model()
    # Create new model
    new_model = tf.keras.models.load_model('sequential.model')
    # Prediction  - predict always take a list
    test = cv2.imread("D:/Data/test/adam_mckay.jpg", cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(test, (90, 90))
    x = np.array(x).reshape(-1, 90, 90, 1)
    predictions = new_model.predict(x)
    print(f"PREDICTION :{np.argmax(predictions[0])}")
    print(f"{x.shape}")
    # x = np.expand_dims(x, axis=0)
    if len(x.shape) == 3:
        plt.imshow(np.squeeze(x), cmap='gray')
    elif len(x.shape) == 2:
        plt.imshow(x, cmap='gray')
    elif len(x.shape) == 4:
        plt.imshow(np.squeeze(x), cmap='gray')
    else:
        print("Higher dimensional data")
    plt.show()


if __name__ == "__main__":
    main()
