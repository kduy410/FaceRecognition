import pickle
import random
import sys
import traceback

import numpy as np
from imageio import imwrite
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm
import vgg_model


class Visualize:
    x = "x_train_96_1_shuffle.npy"
    y = "y_train_96_1_shuffle.npy"
    data = "data"
    dataframe = "dataframe.zip"
    file_name = "lfw"
    dimension = "96"
    types = ['embs', 'tenorbytes', 'labels', 'sprites']
    weight_path = "weights/models-DENSEFINAL96/triplet_weights_model-DENSE-FINAL-96-14.hdf5"

    def __init__(self):
        self.x_train = np.load(f"{self.data}/{self.x}")
        self.y_train = np.load(f"{self.data}/{self.y}")
        self.dataframe = pd.read_csv(f"{self.data}/{self.dataframe}")
        print(f"x-shape:{np.shape(self.x_train)}")
        print(f"y-shape:{np.shape(self.y_train)}")
        print(f"Data frame\n{self.dataframe.head()}")
        model = vgg_model.deep_rank_model(input_shape=self.x_train.shape[1:])
        model.summary()
        model.load_weights(f'{self.weight_path}')
        self.model = model

    def create_embeddings(self, data, name=None):
        embs96 = []
        for x in tqdm(data):
            image = x / 255.
            image = np.expand_dims(image, axis=0)
            emb96 = self.model.predict([image, image, image])
            embs96.append(emb96[0])
            del image
        print(f"EMBS-SHAPE:{np.shape(embs96)}")
        embs96 = np.array(embs96)
        np.save(f"{self.data}/{self.file_name}-{self.dimension}-{self.types[0]}{str(name)}", embs96)
        embs96.tofile(f"{self.data}/{self.file_name}-{self.dimension}-{self.types[1]}{str(name)}.bytes")

    def create_tensor_labels(self, y_train, name=None):
        with open(f'{self.data}/{self.file_name}-{self.dimension}-{self.types[2]}{str(name)}.tsv', 'w') as f:
            for label in tqdm(y_train):
                f.write(str(label) + '\n')

    def images_to_sprite(self, data):
        """
        Creates the sprite image
        :param data: [batch_size, height, weight, n_channel]
        :return data: Sprited image::[height, weight, n_channel]
        """
        if len(data.shape) == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))

        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=0)

        data = data.reshape((n, n) + data.shape[1:]).transpose(
            (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
        )

        data = data.reshape(
            (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data

    def to_sprites(self, data, name=None):
        simg = self.images_to_sprite(data)
        imwrite(f'{self.data}/{self.file_name}-{self.dimension}-{self.types[3]}{str(name)}.png', np.squeeze(simg))
        print("SPRITES-SAVED!!!")

    def generate_random_sample(self, total=5000):
        ids = random.sample(range(0, self.y_train.shape[0]), total)
        x_train = self.x_train[ids]
        y_train = self.y_train[ids]
        self.create_embeddings(x_train, name=f"-{total}")
        self.create_tensor_labels(y_train=y_train, name=f"-{total}")
        self.to_sprites(x_train)

    def generate_sample(self, name):
        try:
            # row x col
            label = self.dataframe[self.dataframe['name'] == name].iloc[0, 1]
        except IndexError as ie:
            print(f"{name} not found!")
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            return

        ids = []
        for i in tqdm(self.y_train):
            if self.y_train[i] == label:
                ids.append(i)
        print(ids)
        x_train = self.x_train[ids]
        y_train = self.y_train[ids]

        self.create_embeddings(x_train, name=f"-{name}")
        self.create_tensor_labels(y_train=y_train, name=f"-{name}")
        self.to_sprites(x_train, name=f"-{name}")
