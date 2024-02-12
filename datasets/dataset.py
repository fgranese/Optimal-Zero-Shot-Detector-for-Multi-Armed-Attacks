import os
import sys
import inspect
import numpy as np
from art.utils import load_cifar10
from utils.utils_general import from_numpy_to_dataloader

CIFAR10 = 'CIFAR10'
SVHN = 'SVHN'

def prep_folder(path: str, to_file: bool = False):
    if to_file:
        path_to_list_of_strings = path.split("/")
        if len(path_to_list_of_strings) < 2:
            sys.exit("Illegal path to file in function " + inspect.stack()[0][3])
        tmp = ""
        for i in range(len(path_to_list_of_strings) - 1):
            tmp += path_to_list_of_strings[i] + "/"
        path = tmp
    os.makedirs(path, exist_ok=True)

def load_svhn_data():
    from subprocess import call
    import scipy.io as sio

    if not os.path.isfile("data/svhn/train_32x32.mat"):
        os.makedirs("data/svhn/", exist_ok=True)
        print('Downloading SVHN train set...')
        call(
            "curl -o data/svhn/train_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            shell=True
        )
    if not os.path.isfile("data/svhn/test_32x32.mat"):
        os.makedirs("data/svhn/", exist_ok=True)
        print('Downloading SVHN test set...')
        call(
            "curl -o data/svhn/test_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            shell=True
        )
    train = sio.loadmat('data/svhn/train_32x32.mat')
    test = sio.loadmat('data/svhn/test_32x32.mat')
    x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
    y_train = np.reshape(train['y'], (-1,))
    y_test = np.reshape(test['y'], (-1,))
    np.place(y_train, y_train == 10, 0)
    np.place(y_test, y_test == 10, 0)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (73257, 32, 32, 3))
    x_test = np.reshape(x_test, (26032, 32, 32, 3))

    min = x_test.min()
    max = x_test.max()

    return (x_train, y_train), (x_test, y_test), min, max

def get_dataloader_from_dataset_name(dataset_name: str, batch_size: int, train: bool, shuffle=False):
    dataset_name = dataset_name.upper()
    if dataset_name == CIFAR10:
        dataloader = get_CIFAR10(batch_size=batch_size, train=train, shuffle=shuffle)
    elif dataset_name == SVHN:
        dataloader = get_SVHN(batch_size=batch_size, train=train, shuffle=shuffle)
    else:
        sys.exit('Requested dataset not available.')
    return dataloader

def get_CIFAR10(batch_size: int, train=False, shuffle=False):

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    y_test = np.argmax(y_test, axis=1)
    y_train = np.argmax(y_train, axis=1)

    if train:
        return from_numpy_to_dataloader(X=x_train, y=y_train, batch_size=batch_size, shuffle=shuffle)
    else:
        return from_numpy_to_dataloader(X=x_test, y=y_test, batch_size=batch_size, shuffle=shuffle)

def get_SVHN(batch_size: int, train=False, shuffle=False):

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_svhn_data()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32) / 255.
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32) / 255.

    if train:
        return from_numpy_to_dataloader(X=x_train, y=y_train, batch_size=batch_size, shuffle=shuffle)
    else:
        return from_numpy_to_dataloader(X=x_test, y=y_test, batch_size=batch_size, shuffle=shuffle)


def get_dataloader(data_name: str, train: bool, batch_size=1, shuffle=False, *args, **kwargs):
    """Returns a dataloader given a dataset"""
    return get_dataloader_from_dataset_name(dataset_name=data_name, batch_size=batch_size, shuffle=shuffle, train=train)

