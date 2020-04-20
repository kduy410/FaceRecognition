import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import model_from_json, Model, load_model
from sklearn.decomposition import PCA, IncrementalPCA

import vgg_model


def main(mode='pca'):
    X_train = np.load('x_train_96_1_shuffle.npy')
    y_train = np.load('y_train_96_1_shuffle.npy')
    print(X_train.shape[1:])
    # X_train = X_train.reshape(X_train.shape[0], 28, 28, 3)
    X_train = X_train.astype('float32')
    X_train /= 255

    # with open('config.json') as f:
    #     config = f.read()
    # model = model_from_json(config)
    model = vgg_model.deep_rank_model((96, 96, 3))
    model.load_weights('weights/models-DENSEFINAL96/triplet_weights_model-DENSE-FINAL-96-144-5.hdf5')

    model.compile(optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True),
                  loss=vgg_model._loss_tensor)
    new_model = Model(model.inputs, model.layers[-3].output)

    new_model.set_weights(model.get_weights())

    embs_4096 = new_model.predict(X_train)
    if mode == 'pca':
        pca = PCA(n_components=96)
        embs_128 = pca.fit_transform(embs_4096)
        with open('embs_96D.pkl', 'wb') as f:
            pickle.dump(embs_128, f)
        embs_128.tofile('data_tensor.bytes')
    elif mode == 'ipca':
        # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA  # noqa
        ipca = IncrementalPCA(n_components=128, batch_size=200)
        embs_128 = ipca.fit_transform(embs_4096)
        with open('embs_96D_2.pkl', 'wb') as f:
            pickle.dump(embs_128, f)
        embs_128.tofile('data_tensor_2.bytes')
    else:
        raise NotImplementedError("Mode must be set")


if __name__ == '__main__':
    main()
