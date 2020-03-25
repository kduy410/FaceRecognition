import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from tqdm import tqdm

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = tqdm(mnist.load_data())

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)  # 60000 example of (28,28) image or 28x28 => sequence of 28 rows of 28 pixels per rows
print(x_train[0].shape)

model = Sequential()
# First layer
model.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))  # 20% dropout
# Another layer
model.add(LSTM(128))
model.add(Dropout(0.2))  # 20% dropout
# Dense layer
model.add(Dense(32, activation='relu'))  # 20% dropout
model.add(Dropout(0.2))  # 20% dropout
# Final Dense layer
model.add(Dense(10, activation='softmax'))
# To do compile, must have optimizers
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

