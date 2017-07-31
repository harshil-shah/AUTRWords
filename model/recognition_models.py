import theano
import theano.tensor as T
from lasagne.layers import DenseLayer, ElemwiseSumLayer, get_all_layers, get_all_param_values, get_all_params, \
    get_output, InputLayer, LSTMLayer, NonlinearityLayer, set_all_param_values
from lasagne.nonlinearities import linear
from nn.nonlinearities import elu_plus_one


class RecModel(object):

    def __init__(self, z_dim, max_length, vocab_size, dist_z):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.dist_z = dist_z()

        self.mean_nn, self.cov_nn = self.nn_fn()

    def nn_fn(self):

        raise NotImplementedError()

    def get_means_and_covs(self, X, X_embedded):

        mask = T.switch(T.lt(X, 0), 0, 1)  # N * max(L)

        X_embedded *= T.shape_padright(mask)

        # X_embedded = theano.printing.Print('X_embedded')(X_embedded)

        means = get_output(self.mean_nn, X_embedded)  # N * dim(z)
        covs = get_output(self.cov_nn, X_embedded)  # N * dim(z)

        # means = theano.printing.Print('means')(means)
        # covs = theano.printing.Print('covs')(covs)

        return means, covs

    def get_samples(self, X, X_embedded, num_samples, means_only=False):
        """
        :param X: N * max(L) matrix
        :param X_embedded: N * max(L) * D tensor
        :param num_samples: int
        :param means_only: bool

        :return samples: (S*N) * dim(z) matrix
        """

        means, covs = self.get_means_and_covs(X, X_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        # samples = theano.printing.Print('samples')(samples)

        return samples

    def log_q_z(self, z, X, X_embedded):
        """
        :param z: (S*N) * dim(z) matrix
        :param X: N * max(L) * D tensor

        :return:
        """

        N = X.shape[0]
        S = T.cast(z.shape[0] / N, 'int32')

        means, covs = self.get_means_and_covs(X, X_embedded)

        means = T.tile(means, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        covs = T.tile(covs, [S] + [1]*(means.ndim - 1))  # (S*N) * dim(z)

        return self.dist_z.log_density(z, [means, covs])

    def kl_std_gaussian(self, X, X_embedded):
        """
        :param X: N * max(L) * D tensor

        :return kl: N length vector
        """

        means, covs = self.get_means_and_covs(X, X_embedded)

        log_covs = -0.5 * T.sum(T.log(covs), axis=range(1, means.ndim))
        # log_covs = theano.printing.Print('kl_log_covs')(log_covs)
        minus_covs = -0.5 * -T.sum(covs, axis=range(1, means.ndim))
        # minus_covs = theano.printing.Print('kl_minus_covs')(minus_covs)
        minus_means_sq = -0.5 * -T.sum((means**2), axis=range(1, means.ndim))
        # minus_means_sq = theano.printing.Print('kl_minus_means_sq')(minus_means_sq)

        kl = -0.5 + log_covs + minus_covs + minus_means_sq

        # kl = -0.5 * T.sum(1. + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return kl

    def get_params(self):

        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecMLP(RecModel):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):

        self.nn_depth = nn_kwargs['depth']
        self.nn_hid_units = nn_kwargs['hid_units']
        self.nn_hid_nonlinearity = nn_kwargs['hid_nonlinearity']

        super(RecMLP, self).__init__(z_dim, max_length, vocab_size, dist_z)

    # def nn_fn(self):
    #
    #     l_in = InputLayer((None, self.max_length, self.vocab_size))
    #
    #     l_current = DenseLayer(l_in, num_units=self.nn_hid_units, nonlinearity=None, b=None)
    #
    #     skip_layers = [l_current]
    #
    #     for h in range(self.nn_depth):
    #
    #         l_h_x = DenseLayer(l_in, num_units=self.nn_hid_units, nonlinearity=None, b=None)
    #         l_h_h = DenseLayer(l_current, num_units=self.nn_hid_units, nonlinearity=None, b=None)
    #
    #         l_sum = NonlinearityLayer(ElemwiseSumLayer([l_h_x, l_h_h]), nonlinearity=self.nn_hid_nonlinearity)
    #
    #         skip_layers.append(l_sum)
    #
    #         l_current = l_sum
    #
    #     l_skip_sum = ElemwiseSumLayer(skip_layers)
    #
    #     mean_nn = DenseLayer(l_skip_sum, num_units=self.z_dim, nonlinearity=linear, b=None)
    #
    #     cov_nn = DenseLayer(l_skip_sum, num_units=self.z_dim, nonlinearity=elu_plus_one, b=None)
    #
    #     return mean_nn, cov_nn

    def nn_fn(self):

        l_in = InputLayer((None, self.max_length, self.vocab_size))

        l_current = l_in

        for h in range(self.nn_depth):

            l_current = DenseLayer(l_current, num_units=self.nn_hid_units, nonlinearity=self.nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_current, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_current, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn


class RecRNN(RecModel):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):

        self.nn_rnn_hid_dim = nn_kwargs['rnn_hid_dim']
        self.nn_final_depth = nn_kwargs['final_depth']
        self.nn_final_hid_units = nn_kwargs['final_hid_units']
        self.nn_final_hid_nonlinearity = nn_kwargs['final_hid_nonlinearity']

        super(RecRNN, self).__init__(z_dim, max_length, vocab_size, dist_z)

        self.rnn = self.rnn_fn()

    def rnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.vocab_size))

        l_mask = InputLayer((None, self.max_length))

        l_final = LSTMLayer(l_in, num_units=self.nn_rnn_hid_dim, mask_input=l_mask, only_return_final=True)

        return l_final

    def nn_fn(self):

        l_in = InputLayer((None, self.nn_rnn_hid_dim))

        l_prev = l_in

        for h in range(self.nn_final_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_final_hid_units, nonlinearity=self.nn_final_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_means_and_covs(self, X, X_embedded):

        mask = T.switch(T.lt(X, 0), 0, 1)  # N * max(L)
        hid = self.rnn.get_output_for([X_embedded, mask])  # N * dim(z)

        means = get_output(self.mean_nn, hid)  # N * dim(z)
        covs = get_output(self.cov_nn, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(self.rnn, trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(self.rnn)
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(self.rnn, rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)
