from fashion_mnist_input import *
from resnet import *

def generate_augment_train_batch(train_data, train_labels, train_batch_size):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
    batch_data = train_data[offset:offset+train_batch_size, ...]
    batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

    batch_data = whitening_image(batch_data)
    batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

    return batch_data, batch_label

all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
train_batch_data, train_batch_labels = generate_augment_train_batch(all_data, all_labels,
                                                                        FLAGS.train_batch_size)
print(train_batch_labels)
