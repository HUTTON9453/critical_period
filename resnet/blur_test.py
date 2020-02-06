import matplotlib.pyplot as plt
import tensorflow as tf
from cifar10_input import *
import pickle

full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_1'
#data, label = prepare_train_data(0)
fo = open(full_data_dir, 'rb')
dicts = pickle.load(fo, encoding = 'latin1')
fo.close()
data = dicts['data']
label = np.array(dicts['labels'])

num_data = len(label)

data = data.reshape((num_data,32*32,3), order='F')
data = data.reshape((num_data,32,32,3))
data = data/255

with tf.Session() as sess:
    data = tf.image.convert_image_dtype(data, dtype = tf.float32)
    resize_0 = tf.image.resize(data, (8, 8))
    down = resize_0.eval()
    resize_1 = tf.image.resize(down, (32, 32))

    print(resize_0.get_shape)

    print(data[0])
    plt.imshow(resize_1.eval()[0])
    plt.savefig('blur_test_img.png')


