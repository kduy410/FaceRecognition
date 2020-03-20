import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
import pandas as pd
from tqdm import tqdm

# Load model train sẵn lên
model_path = './model/facenet_keras.h5'
model = load_model(model_path)


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


def load_and_align_images(filepaths, margin, image_size=160):
    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)
        aligned = resize(img, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)

    return np.array(aligned_images)


# Hàm tính embedding, trong code gốc để l2_normalize nên mình cũng để nguyên vậy
def calc_embs(filepaths, margin=10, batch_size=512):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = prewhiten(load_and_align_images(filepaths[start:start + batch_size], margin))
        pd.append(model.predict_on_batch(aligned_images))
    embs = l2_normalize(np.concatenate(pd))

    return embs


# Load các file metadata
test_path = "D:/Data/test/adam-sandler-vertical.jpg"
train_path = 'D:/Data/images/faces/Adam_Sandler/'

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('sample_submission.csv')

# Bây giờ ta tính embeddings cho tất cả các ảnh trong tập train cũng như tập test
train_embs = calc_embs([os.path.join(train_path, f) for f in train_df.image.values])
np.save("train_embs.npy", train_embs)

test_embs = calc_embs([os.path.join(test_path, f) for f in test_df.image.values])
np.save("test_embs.npy", test_embs)

# Lọc ra tập các index trong file train.csv chứa cùng 1 label (từ 0 - 999), tiện cho tính toán bên dưới
# indices which belong to each label
label2idx = []

for i in tqdm(range(1000)):
    label2idx.append(np.asarray(train_df[train_df.label == i].index))

# Để chèn class 1000 vào dự đoán, mình dùng 1 phương pháp đơn giản là dùng ngưỡng.
# Đầu tiên mình sẽ vẽ phân phối xác suất của 2 biến:
# Khoảng cách euclide giữa 2 ảnh nếu chúng thuộc cùng một mặt.
# Khoảng cách euclide giữa 2 ảnh nếu chúng thuộc 2 mặt khác nhau.
# Mục đích là tìm 1 khoảng cách thích hợp để nếu khoảng cách từ 1 ảnh đến tất cả các ảnh đều cao hơn nó thì sẽ khẳng định nó là unkown vậy thôi.

import matplotlib.pyplot as plt

match_distances = []
for i in range(1000):
    ids = label2idx[i]
    distances = []
    for j in range(len(ids) - 1):
        for k in range(j + 1, len(ids)):
            distances.append(distance.euclidean(train_embs[ids[j]], train_embs[ids[k]]))
    match_distances.extend(distances)

unmatch_distances = []
for i in range(1000):
    ids = label2idx[i]
    distances = []
    for j in range(5):
        idx = np.random.randint(train_embs.shape[0])
        while idx in label2idx[i]:
            idx = np.random.randint(train_embs.shape[0])
        distances.append(distance.euclidean(train_embs[ids[np.random.randint(len(ids))]], train_embs[idx]))
    unmatch_distances.extend(distances)

_, _, _ = plt.hist(match_distances, bins=100)
_, _, _ = plt.hist(unmatch_distances, bins=100, fc=(1, 0, 0, 0.5))

# Mình sẽ chọn chỗ giao nhau giữa 2 thằng tức là tầm 1.1
threshold = 1.1

# Tính toán khoảng cách giữa 2 tập và đưa ra dự đoán.
for i in tqdm(range(len(test_df.image))):
    distances = []
    for j in range(1000):
        distances.append(np.min([distance.euclidean(test_embs[i], train_embs[k]) for k in label2idx[j]]))
    distances.append(threshold)
    test_df.loc[i].label = ' '.join([str(p) for p in np.argsort(distances)[:5]])

test_df.to_csv("sub.csv", index=False)
