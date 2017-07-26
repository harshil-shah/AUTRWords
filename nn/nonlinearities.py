import theano.tensor as T
from lasagne.nonlinearities import elu


def elu_plus_one(x):

    return elu(x) + 1. + 1.e-5
