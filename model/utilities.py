import theano.tensor as T


def last_d_softmax(x):

    e_x = T.exp(x - x.max(axis=-1, keepdims=True))
    out = e_x / e_x.sum(axis=-1, keepdims=True)

    return out
