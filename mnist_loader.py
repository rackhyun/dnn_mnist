import numpy as np
import os
import urllib
import gzip
import matplotlib.pyplot as plt


def load_mnist():
    url_tr_dat = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_tr_lab = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    url_ts_dat = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_ts_lab = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    if not os.path.exists('./mnist-batches-py'):
        os.mkdir('mnist-batches-py')

        urllib.request.urlretrieve(url_tr_dat, './mnist-batches-py/train-images-idx3-ubyte.gz')
        urllib.request.urlretrieve(url_tr_lab, './mnist-batches-py/train-labels-idx1-ubyte.gz')
        urllib.request.urlretrieve(url_ts_dat, './mnist-batches-py/t10k-images-idx3-ubyte.gz')
        urllib.request.urlretrieve(url_ts_lab, './mnist-batches-py/t10k-labels-idx1-ubyte.gz')

    X_train_f = gzip.open('./mnist-batches-py/train-images-idx3-ubyte.gz', 'rb')
    y_train_f = gzip.open('./mnist-batches-py/train-labels-idx1-ubyte.gz', 'rb')
    X_test_f = gzip.open('./mnist-batches-py/t10k-images-idx3-ubyte.gz', 'rb')
    y_test_f = gzip.open('./mnist-batches-py/t10k-labels-idx1-ubyte.gz', 'rb')

    s = X_train_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    X_train = loaded[16:].reshape((60000, 1, 28, 28)).astype(float)

    s = y_train_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    y_train = loaded[8:].reshape((60000,)).astype('uint8')

    s = X_test_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    X_test = loaded[16:].reshape((10000, 1, 28, 28)).astype('uint8')

    s = y_test_f.read()
    loaded = np.frombuffer(s, dtype=np.uint8)
    y_test = loaded[8:].reshape((10000,)).astype('uint8')

    X_train_f.close()
    y_train_f.close()
    X_test_f.close()
    y_test_f.close()

    return X_train, y_train, X_test, y_test
