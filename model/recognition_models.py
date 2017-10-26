import theano
import theano.tensor as T
from lasagne.layers import DenseLayer, get_all_layers, get_all_param_values, get_all_params, get_output, InputLayer, \
    LSTMLayer, set_all_param_values
from lasagne.nonlinearities import linear
from nn.nonlinearities import elu_plus_one

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams(1234)


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

        means = get_output(self.mean_nn, X_embedded)  # N * dim(z)
        covs = get_output(self.cov_nn, X_embedded)  # N * dim(z)

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

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return kl

    def get_samples_and_kl_std_gaussian(self, X, X_embedded, num_samples, means_only=False):

        means, covs = self.get_means_and_covs(X, X_embedded)

        if means_only:
            samples = T.tile(means, [num_samples] + [1]*(means.ndim - 1))  # (S*N) * dim(z)
        else:
            samples = self.dist_z.get_samples(num_samples, [means, covs])  # (S*N) * dim(z)

        kl = -0.5 * T.sum(T.ones_like(means) + T.log(covs) - covs - (means**2), axis=range(1, means.ndim))

        return samples, kl

    def summarise_z(self, all_x, all_x_embedded):

        def step(all_x_s, all_x_embedded_s):

            means, covs = self.get_means_and_covs(all_x_s, all_x_embedded_s)  # N * dim(z) and N * dim(z)

            precs = 1. / covs  # N * dim(z)

            weighted_means = means * precs  # N * dim(z)

            return precs, weighted_means

        ([all_precs, all_weighted_means], _) = theano.scan(step,
                                                           sequences=[all_x.dimshuffle((1, 0, 2)),
                                                                      all_x_embedded.dimshuffle((1, 0, 2, 3))],
                                                           )

        total_precision = T.sum(all_precs, axis=0)  # N * dim(z)

        summary_mean = T.sum(all_weighted_means, axis=0) / total_precision  # N * dim(z)

        return summary_mean

    def get_params(self):

        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return nn_params

    def get_param_values(self):

        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [nn_params_vals]

    def set_param_values(self, param_values):

        [nn_params_vals] = param_values

        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)


class RecRNN(RecModel):

    def __init__(self, z_dim, max_length, vocab_size, dist_z, nn_kwargs):

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.nn_nn_depth = nn_kwargs['nn_depth']
        self.nn_nn_hid_units = nn_kwargs['nn_hid_units']
        self.nn_nn_hid_nonlinearity = nn_kwargs['nn_hid_nonlinearity']

        super().__init__(z_dim, max_length, vocab_size, dist_z)

        self.rnn = self.rnn_fn()

    def rnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.vocab_size))

        l_mask = InputLayer((None, self.max_length))

        l_prev = l_in

        all_layers = []

        for h in range(self.nn_rnn_depth):

            l_prev = LSTMLayer(l_prev, num_units=self.nn_rnn_hid_units, mask_input=l_mask)

            all_layers.append(l_prev)

        return all_layers

    def nn_fn(self):

        l_in = InputLayer((None, self.nn_rnn_depth * self.nn_rnn_hid_units))

        l_prev = l_in

        for h in range(self.nn_nn_depth):

            l_prev = DenseLayer(l_prev, num_units=self.nn_nn_hid_units, nonlinearity=self.nn_nn_hid_nonlinearity)

        mean_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=linear)

        cov_nn = DenseLayer(l_prev, num_units=self.z_dim, nonlinearity=elu_plus_one)

        return mean_nn, cov_nn

    def get_hid(self, X, X_embedded):

        mask = T.switch(T.lt(X, 0), 0, 1)  # N * max(L)

        h_prev = X_embedded  # N * max(L) * E

        all_h = []

        for h in range(len(self.rnn)):

            h_prev = self.rnn[h].get_output_for([h_prev, mask])  # N * max(L) * dim(hid)

            all_h.append(h_prev[:, -1])

        hid = T.concatenate(all_h, axis=-1)

        return hid

    def get_means_and_covs(self, X, X_embedded):

        hid = self.get_hid(X, X_embedded)  # N * (depth*dim(hid))

        means = get_output(self.mean_nn, hid)  # N * dim(z)
        covs = get_output(self.cov_nn, hid)  # N * dim(z)

        return means, covs

    def get_params(self):

        rnn_params = get_all_params(get_all_layers(self.rnn), trainable=True)
        nn_params = get_all_params(get_all_layers([self.mean_nn, self.cov_nn]), trainable=True)

        return rnn_params + nn_params

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(get_all_layers(self.rnn))
        nn_params_vals = get_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]))

        return [rnn_params_vals, nn_params_vals]

    def set_param_values(self, param_values):

        [rnn_params_vals, nn_params_vals] = param_values

        set_all_param_values(get_all_layers(self.rnn), rnn_params_vals)
        set_all_param_values(get_all_layers([self.mean_nn, self.cov_nn]), nn_params_vals)
