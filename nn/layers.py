import theano.tensor as T
from lasagne.theano_extensions import conv
from lasagne import init, nonlinearities
from lasagne.layers import Conv1DLayer, Layer, PadLayer


class RepeatLayer(Layer):

    def __init__(self, incoming, repeats, axis, ndim, name=None):

        self.repeats = repeats
        self.axis = axis
        self.ndim = ndim

        super(RepeatLayer, self).__init__(incoming=incoming, name=name)

    def get_output_shape_for(self, input_shape):

        output_shape = list(input_shape)
        output_shape[self.axis] *= self.repeats

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):

        repeat_pattern = [1] * self.ndim
        repeat_pattern[self.axis] = self.repeats

        return T.tile(input, repeat_pattern)
