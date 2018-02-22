import os
import numpy as np
from sklearn.utils import shuffle


# #############################################################################
# Load data

def load_and_split_data(datafile):
    data = np.load(datafile)
    X, y, = data['Features'], data['Targets']
    X, y = shuffle(X, y, random_state=13)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    return (X_train, y_train, X_test, y_test)


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

def get_ht_defect_data():
    X_train, y_train, X_test, y_test = zip(load_and_split_data(os.path.join(path, "quaternary_training_data.npz")),
                                           load_and_split_data(os.path.join(path, "ternary_training_data.npz")))
    X_train, y_train, X_test, y_test = map(np.concatenate, [X_train, y_train, X_test, y_test])
    return X_train, y_train, X_test, y_test

