from keras import applications, Input, Model
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Lambda, Conv2D, MaxPool2D, Flatten, concatenate

import numpy as np

batch_size = 24
_EPSILON = K.epsilon()


def _loss_tensor(y_true, y_pred):
    global _EPSILON, batch_size
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


def image_batch_generator(images, labels, batch_size):
    labels = np.array(labels)
    while True:
        batch_paths = np.random.choice(a=len(images), size=batch_size // 3)
        input_1 = []

        for i in batch_paths:
            pos = np.where(labels == labels[i])[0]
            neg = np.where(labels != labels[i])[0]

            if len(pos) is not 1:
                j = np.random.choice(pos)
                while j == i:
                    j = np.random.choice(pos)
            else:
                j = i

            k = np.random.choice(neg)
            while k == i:
                k = np.random.choice(neg)

            input_1.append(images[i])
            input_1.append(images[j])
            input_1.append(images[k])

        input_1 = np.array(input_1)
        input = [input_1, input_1, input_1]
        yield input, np.zeros((batch_size,))


def convnet_model_(input_shape):
    vgg_model = applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Lambda(lambda x_: K.l2_normalize(x, axis=1))(x)
    #     x = Lambda(K.l2_normalize)(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model


def deep_rank_model(input_shape):
    convnet_model = convnet_model_(input_shape)

    first_input = Input(shape=input_shape)
    first_conv = Conv2D(96, kernel_size=(8, 8), strides=(16, 16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

    second_input = Input(shape=input_shape)
    second_conv = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7, 7), strides=(4, 4), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    emb = Dense(128)(emb)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model
