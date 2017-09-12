import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import ConcatLayer, DenseLayer, ElemwiseSumLayer, get_all_param_values, get_all_params, get_output, \
    InputLayer, LSTMLayer, NonlinearityLayer, RecurrentLayer, set_all_param_values
from lasagne.nonlinearities import sigmoid, tanh

from .utilities import last_d_softmax

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

random = RandomStreams()


class GenStanfordWords(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.nn_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.rnn = self.rnn_fn()

        self.W_z = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.embedding_dim))))

    def rnn_fn(self):

        l_in = InputLayer((None, self.max_length, self.embedding_dim))

        l_current = l_in

        for h in range(self.nn_rnn_depth):

            l_current = LSTMLayer(l_current, num_units=self.nn_rnn_hid_units, nonlinearity=self.nn_rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_current, num_units=self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=None)

        return l_out

    def log_p_z(self, z):
        """
        :param z: (S*N) * dim(z) matrix

        :return log_p_z: (S*N) length vector
        """

        log_p_z = self.dist_z.log_density(z)

        return log_p_z

    def get_probs(self, x, x_dropped, z, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * E tensor
        """

        z_start = T.dot(z, self.W_z)  # N * E

        x_dropped_pre_padded = T.concatenate([T.shape_padaxis(z_start, 1), x_dropped], axis=1)[:, :-1]  # (S*N) * max(L)
        # * E

        hiddens = get_output(self.rnn, x_dropped_pre_padded)  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * hiddens, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(hiddens, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_x(self, x, x_embedded, x_embedded_dropped, z, all_embeddings):
        """
        :param x: N * max(L) tensor
        :param x_embedded: N * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param all_embeddings: D * E matrix

        :return log_p_x: (S*N) length vector
        """

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * max(L)
        x_rep_padding_mask = T.switch(T.lt(x_rep, 0), 0, 1)  # (S*N) * max(L)

        x_embedded_rep = T.tile(x_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_embedded_dropped_rep = T.tile(x_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        probs = self.get_probs(x_embedded_rep, x_embedded_dropped_rep, z, all_embeddings, mode='true')  # (S*N) * max(L)
        probs += T.cast(1.e-15, 'float32')  # (S*N) * max(L)

        log_p_x = T.sum(x_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        L = T.sum(x_rep_padding_mask, axis=1)  # (S*N)

        perplexity = T.exp((-1./L) * log_p_x)  # (S*N)

        return log_p_x, perplexity

    def beam_search(self, z, all_embeddings, beam_size):

        N = z.shape[0]

        z_start = T.repeat(T.dot(z, self.W_z), beam_size, axis=0)  # (N*B) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        best_edges_0 = -T.ones((N, self.vocab_size))  # N * D
        active_words_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        def step_forward(l, best_scores_lm1, best_edges_lm1, active_words_current, all_embeddings, z_start):

            active_words_embedded = self.embedder(T.cast(active_words_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            active_words_embedded_reshape = T.reshape(active_words_embedded, (N*beam_size, self.max_length,
                                                                              self.embedding_dim))  # (N*B) * max(L) * E

            x_pre_padded = T.concatenate([T.shape_padaxis(z_start, 1), active_words_embedded_reshape], axis=1)[:, :-1]
            # (N*B) * max(L) * E

            target_embeddings = get_output(self.rnn, x_pre_padded)[:, l].reshape((N, beam_size, self.embedding_dim))
            # N * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * B * D

            probs = last_d_softmax(probs_denominators)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_new = T.set_subtensor(active_words_current[:, :, l], active_words_l)  # N * B * max(L)

            best_edges_l_beam_inds = T.argmax(scores, axis=1)  # N * D

            best_edges_l = active_words_current[:, :, l-1][T.repeat(T.arange(N), self.vocab_size),
                                                           best_edges_l_beam_inds.flatten()]
            best_edges_l = best_edges_l.reshape((N, self.vocab_size))  # N * D

            return best_scores_l, T.cast(best_edges_l, 'float32'), T.cast(active_words_new, 'float32')

        ([best_scores, best_edges, active_words], _) = theano.scan(step_forward,
                                                                   sequences=T.arange(self.max_length),
                                                                   outputs_info=[best_scores_0, best_edges_0,
                                                                                 active_words_init],
                                                                   non_sequences=[all_embeddings, z_start]
                                                                   )
        # max(L) * N * B and max(L) * N * D and max(L) * N * B * max(L)

        words_L = active_words[-1, :, -1, -1]  # N

        def step_backward(best_edges_l, words_lp1):

            words_l = best_edges_l[T.arange(N), T.cast(words_lp1, 'int32')]  # N

            return words_l

        words, _ = theano.scan(step_backward,
                               sequences=best_edges[1:][::-1],
                               outputs_info=[words_L],
                               )

        words = words[::-1]  # (max(L)-1) * N

        words = T.concatenate([words, T.shape_padleft(words_L)], axis=0).T  # N * max(L)

        return T.cast(words, 'int32')

    def impute_missing_words(self, z, x_best_guess, missing_words_mask, all_embeddings, beam_size):

        N = z.shape[0]

        z_start = T.repeat(T.dot(z, self.W_z), beam_size, axis=0)  # (N*B) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        def step_forward(l, x_best_guess_l, missing_words_mask_l, best_scores_lm1, active_paths_current, all_embeddings,
                         z_start):

            active_words_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N * B *
            # max(L) * E

            active_words_embedded_reshape = T.reshape(active_words_embedded, (N*beam_size, self.max_length,
                                                                              self.embedding_dim))  # (N*B) * max(L) * E

            x_pre_padded = T.concatenate([T.shape_padaxis(z_start, 1), active_words_embedded_reshape], axis=1)[:, :-1]
            # (N*B) * max(L) * E

            target_embeddings = get_output(self.rnn, x_pre_padded)[:, l].reshape((N, beam_size, self.embedding_dim))
            # N * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * B * D

            probs = last_d_softmax(probs_denominators)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            best_scores_l_known = best_scores_l_all[T.arange(N), x_best_guess_l]  # N

            best_scores_l = T.switch(T.eq(T.shape_padright(missing_words_mask_l), 0.), best_scores_l,
                                     T.shape_padright(best_scores_l_known))  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            active_words_l = T.switch(T.eq(T.shape_padright(missing_words_mask_l), 0.), active_words_l,
                                      T.shape_padright(x_best_guess_l))  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]  # (N*B)
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()]  # (N*B)
            # * max(L)
            best_paths_l = best_paths_l.reshape((N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return best_scores_l, active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=[T.arange(self.max_length), x_best_guess.T,
                                                                  missing_words_mask.T],
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[all_embeddings, z_start]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def generate_text(self, z, all_embeddings):
        """
        :param z: N * dim(z) matrix
        :param all_embeddings: D * E matrix

        :return x: N * max(L) tensor
        """

        N = z.shape[0]

        x_init_sampled = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)
        x_init_argmax = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)

        def step(l, x_prev_sampled, x_prev_argmax, z, all_embeddings):

            x_prev_sampled_embedded = self.embedder(x_prev_sampled, all_embeddings)  # N * max(L) * E

            probs_sampled = self.get_probs(x_prev_sampled_embedded, x_prev_sampled_embedded, z, all_embeddings,
                                           mode='all')  # N * max(L) * D

            x_sampled_one_hot = self.dist_x.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_sampled_l = T.argmax(x_sampled_one_hot, axis=-1).flatten()  # N

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l], x_sampled_l)  # N * max(L)

            #

            x_prev_argmax_embedded = self.embedder(x_prev_argmax, all_embeddings)  # N * max(L) * E

            probs_argmax = self.get_probs(x_prev_argmax_embedded, x_prev_argmax_embedded, z, all_embeddings, mode='all')
            # N * max(L) * D

            x_argmax_l = T.argmax(probs_argmax[:, l], axis=-1)  # N

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_l)  # N * max(L)

            return T.cast(x_current_sampled, 'int32'), T.cast(x_current_argmax, 'int32')

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z, all_embeddings],
                                                     )

        return x_sampled[-1], x_argmax[-1], updates

    def generate_output_prior_fn(self, all_embeddings, num_samples, beam_size, num_time_steps=None):

        z = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax, x_gen_beam, x_gen_beam],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior

    def generate_output_posterior_fn(self, x, z, all_embeddings, beam_size, num_time_steps=None):

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size)

        generate_output_posterior = theano.function(inputs=[x],
                                                    outputs=[z, x_gen_sampled, x_gen_argmax, x_gen_beam],
                                                    updates=updates,
                                                    allow_input_downcast=True
                                                    )

        return generate_output_posterior

    def find_best_matches_fn(self, sentences_in, sentences_eval, sentences_eval_embedded, z, all_embeddings):

        S = sentences_in.shape[0]
        N = sentences_eval.shape[0]

        sentences_eval_rep = T.tile(sentences_eval, (S, 1))  # (S*N) * max(L)
        sentences_eval_embedded_rep = T.tile(sentences_eval_embedded, (S, 1, 1))  # (S*N) * max(L) * E

        z_rep = T.extra_ops.repeat(z, repeats=N, axis=0)  # (S*N) * dim(z)

        log_p_x, _ = self.log_p_x(sentences_eval_rep, sentences_eval_embedded_rep, sentences_eval_embedded_rep, z_rep,
                                  all_embeddings)  # (S*N)

        log_p_x = log_p_x.reshape((S, N))

        find_best_matches = theano.function(inputs=[sentences_in, sentences_eval],
                                            outputs=log_p_x,
                                            allow_input_downcast=True,
                                            on_unused_input='ignore',
                                            )

        return find_best_matches

    def get_params(self):

        rnn_params = get_all_params(self.rnn, trainable=True)

        return rnn_params + [self.W_z]

    def get_param_values(self):

        rnn_params_vals = get_all_param_values(self.rnn)

        W_z_val = self.W_z.get_value()

        return [rnn_params_vals, W_z_val]

    def set_param_values(self, param_values):

        [rnn_params_vals, W_z_val] = param_values

        set_all_param_values(self.rnn, rnn_params_vals)

        self.W_z.set_value(W_z_val)


class GenAUTRWords(object):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedder = embedder

        self.nn_canvas_rnn_depth = nn_kwargs['rnn_depth']
        self.nn_canvas_rnn_hid_units = nn_kwargs['rnn_hid_units']
        self.nn_canvas_rnn_hid_nonlinearity = nn_kwargs['rnn_hid_nonlinearity']
        self.nn_canvas_rnn_time_steps = nn_kwargs['rnn_time_steps']

        self.dist_z = dist_z()
        self.dist_x = dist_x()

        self.canvas_rnn = self.canvas_rnn_fn()

        self.canvas_update_params = self.init_canvas_update_params()

    def init_canvas_update_params(self):

        W_h_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_h_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                     self.embedding_dim))))

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim + 2*self.embedding_dim,
                                                                       self.embedding_dim))))

        canvas_update_params = [W_h_Cg, W_h_Cu, W_x_to_x]

        return canvas_update_params

    def canvas_rnn_fn(self):

        l_in = InputLayer((None, None, self.z_dim))

        l_current = LSTMLayer(l_in, num_units=self.nn_canvas_rnn_hid_units,
                              nonlinearity=self.nn_canvas_rnn_hid_nonlinearity)

        skip_layers = [l_current]

        for h in range(self.nn_canvas_rnn_depth - 1):

            l_h_z = LSTMLayer(l_in, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=None)
            l_h_h = LSTMLayer(l_current, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=None)

            l_sum = NonlinearityLayer(ElemwiseSumLayer([l_h_z, l_h_h]),
                                      nonlinearity=self.nn_canvas_rnn_hid_nonlinearity)

            skip_layers.append(l_sum)

            l_resid = ElemwiseSumLayer([l_current, l_sum])

            l_current = l_resid

        l_skip_sum = ElemwiseSumLayer(skip_layers)

        return l_skip_sum

    def canvas_updater(self, hiddens):
        """
        :param hiddens: N * T * dim(hid) tensor

        :return canvases: T * N * max(L) * E tensor
        """

        canvas_init = T.zeros((hiddens.shape[0], self.max_length, self.embedding_dim))  # N * max(L) * E
        gate_sum_init = T.zeros((hiddens.shape[0], self.max_length))  # N * max(L)

        def step(h_t, canvas_tm1, canvas_gate_sum_tm1, W_h_Cg, W_h_Cu):

            pre_softmax_gate = T.dot(h_t, W_h_Cg)  # N * max(L)

            gate_exp = T.exp(pre_softmax_gate - pre_softmax_gate.max(axis=-1, keepdims=True))  # N * max(L)
            unnormalised_gate = gate_exp * (1. - canvas_gate_sum_tm1)  # N * max(L)
            canvas_gate = unnormalised_gate / T.sum(unnormalised_gate, axis=-1, keepdims=True)  # N * max(L)
            canvas_gate *= (1. - canvas_gate_sum_tm1)  # N * max(L)

            canvas_gate_sum = canvas_gate_sum_tm1 + canvas_gate  # N * max(L)
            canvas_gate_reshape = T.shape_padright(canvas_gate)  # N * max(L) * 1

            canvas_update = T.dot(h_t, W_h_Cu)  # N * E
            canvas_update = T.shape_padaxis(canvas_update, 1)  # N * 1 * E

            canvas_new = ((1. - canvas_gate_reshape) * canvas_tm1) + (canvas_gate_reshape * canvas_update)  # N * max(L)
            # * E

            return T.cast(canvas_new, 'float32'), T.cast(canvas_gate_sum, 'float32')

        ([canvases, canvas_gate_sums], _) = theano.scan(step,
                                                        sequences=[hiddens.dimshuffle((1, 0, 2))],
                                                        outputs_info=[canvas_init, gate_sum_init],
                                                        non_sequences=self.canvas_update_params[:2],
                                                        )

        return canvases[-1], canvas_gate_sums[-1]

    def get_canvases(self, z, num_time_steps=None):
        """
        :param z: N * dim(z) matrix
        :param num_time_steps: int, number of RNN time steps to use

        :return canvases: N * max(L) * E tensor
        :return canvas_gate_sums: N * max(L) matrix
        """

        if num_time_steps is None:
            num_time_steps = self.nn_canvas_rnn_time_steps

        z_rep = T.tile(z.reshape((z.shape[0], 1, z.shape[1])), (1, num_time_steps, 1))  # N * T * dim(z)

        hiddens = get_output(self.canvas_rnn, z_rep)  # N * T * dim(hid)

        canvases, canvas_gate_sums = self.canvas_updater(hiddens)  # N * max(L) * E and N * max(L)

        return canvases, canvas_gate_sums

    def get_probs(self, x, x_dropped, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))  # (S*N) * max(L) * dim(z)

        target_embeddings = T.dot(T.concatenate((z_rep, x_pre_padded, canvases), axis=-1),
                                  self.canvas_update_params[-1])  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def log_p_x(self, x, x_embedded, x_embedded_dropped, z, all_embeddings):
        """
        :param x: N * max(L) tensor
        :param x_embedded: N * max(L) * E tensor
        :param x_embedded_dropped: N * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param all_embeddings: D * E matrix

        :return log_p_x: (S*N) length vector
        """

        S = T.cast(z.shape[0] / x.shape[0], 'int32')

        x_rep = T.tile(x, (S, 1))  # (S*N) * max(L)
        x_rep_padding_mask = T.switch(T.lt(x_rep, 0), 0, 1)  # (S*N) * max(L)

        x_embedded_rep = T.tile(x_embedded, (S, 1, 1))  # (S*N) * max(L) * E
        x_embedded_dropped_rep = T.tile(x_embedded_dropped, (S, 1, 1))  # (S*N) * max(L) * E

        canvases = self.get_canvases(z)[0]

        probs = self.get_probs(x_embedded_rep, x_embedded_dropped_rep, z, canvases, all_embeddings, mode='true')
        # (S*N) * max(L)
        probs += T.cast(1.e-15, 'float32')  # (S*N) * max(L)

        log_p_x = T.sum(x_rep_padding_mask * T.log(probs), axis=-1)  # (S*N)

        L = T.sum(x_rep_padding_mask, axis=1)  # (S*N)

        perplexity = T.exp((-1./L) * log_p_x)  # (S*N)

        return log_p_x, perplexity

    def beam_search(self, z, all_embeddings, beam_size, num_time_steps=None):

        N = z.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, beam_size, 1))  # N * B * dim(z)

        canvases = self.get_canvases(z, num_time_steps)[0]  # N * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        best_edges_0 = T.zeros((N, self.vocab_size))  # N * D
        active_words_0 = -T.ones((N, beam_size))  # N * B

        def step_forward(canvases_l, best_scores_lm1, best_edges_lm1, active_words_lm1, W_x_to_x, all_embeddings,
                         z_rep):

            active_words_lm1_embedded = self.embedder(T.cast(active_words_lm1, 'int32'), all_embeddings)  # N * B * E

            target_embeddings = T.dot(T.concatenate((z_rep, active_words_lm1_embedded,
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * B * D

            probs = last_d_softmax(probs_denominators)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            best_edges_l_beam_inds = T.argmax(scores, axis=1)  # N * D

            best_edges_l = active_words_lm1[T.repeat(T.arange(N), self.vocab_size), best_edges_l_beam_inds.flatten()]
            best_edges_l = best_edges_l.reshape((N, self.vocab_size))  # N * D

            return best_scores_l, T.cast(best_edges_l, 'float32'), T.cast(active_words_l, 'float32')

        ([best_scores, best_edges, active_words], _) = theano.scan(step_forward,
                                                                   sequences=[canvases.dimshuffle((1, 0, 2))],
                                                                   outputs_info=[best_scores_0, best_edges_0,
                                                                                 active_words_0],
                                                                   non_sequences=[self.canvas_update_params[-1],
                                                                                  all_embeddings, z_rep]
                                                                   )
        # max(L) * N * B and max(L) * N * D and max(L) * N * B

        words_L = active_words[-1, :, -1]  # N

        def step_backward(best_edges_l, words_lp1):

            words_l = best_edges_l[T.arange(N), T.cast(words_lp1, 'int32')]  # N

            return words_l

        words, _ = theano.scan(step_backward,
                               sequences=best_edges[1:][::-1],
                               outputs_info=[words_L],
                               )

        words = words[::-1]  # (max(L)-1) * N

        words = T.concatenate([words, T.shape_padleft(words_L)], axis=0).T  # N * max(L)

        return T.cast(words, 'int32')

    def generate_text(self, z, all_embeddings, num_time_steps=None):

        N = z.shape[0]

        canvases = self.get_canvases(z, num_time_steps)[0]

        x_init_sampled = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)
        x_init_argmax = T.cast(-1, 'int32') * T.ones((N, self.max_length), 'int32')  # N * max(L)

        def step(l, x_prev_sampled, x_prev_argmax, z, all_embeddings):

            x_prev_sampled_embedded = self.embedder(x_prev_sampled, all_embeddings)  # N * max(L) * E

            probs_sampled = self.get_probs(x_prev_sampled_embedded, x_prev_sampled_embedded, z, canvases,
                                           all_embeddings, mode='all')  # N * max(L) * D

            x_sampled_one_hot = self.dist_x.get_samples([T.shape_padaxis(probs_sampled[:, l], 1)])  # N * 1 * D

            x_sampled_l = T.argmax(x_sampled_one_hot, axis=-1).flatten()  # N

            x_current_sampled = T.set_subtensor(x_prev_sampled[:, l], x_sampled_l)  # N * max(L)

            #

            x_prev_argmax_embedded = self.embedder(x_prev_argmax, all_embeddings)  # N * max(L) * E

            probs_argmax = self.get_probs(x_prev_argmax_embedded, x_prev_argmax_embedded, z, canvases, all_embeddings,
                                          mode='all')  # N * max(L) * D

            x_argmax_l = T.argmax(probs_argmax[:, l], axis=-1)  # N

            x_current_argmax = T.set_subtensor(x_prev_argmax[:, l], x_argmax_l)  # N * max(L)

            return T.cast(x_current_sampled, 'int32'), T.cast(x_current_argmax, 'int32')

        (x_sampled, x_argmax), updates = theano.scan(step,
                                                     sequences=[T.arange(self.max_length)],
                                                     outputs_info=[x_init_sampled, x_init_argmax],
                                                     non_sequences=[z, all_embeddings],
                                                     )

        return x_sampled[-1], x_argmax[-1], updates

    def generate_output_prior_fn(self, all_embeddings, num_samples, beam_size, num_time_steps=None):

        z = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        _, attention = self.get_canvases(z, num_time_steps)

        generate_output_prior = theano.function(inputs=[],
                                                outputs=[z, x_gen_sampled, x_gen_argmax, x_gen_beam, attention],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_output_prior

    def generate_output_posterior_fn(self, x, z, all_embeddings, beam_size, num_time_steps=None):

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        generate_output_posterior = theano.function(inputs=[x],
                                                    outputs=[z, x_gen_sampled, x_gen_argmax, x_gen_beam],
                                                    updates=updates,
                                                    allow_input_downcast=True
                                                    )

        return generate_output_posterior

    def generate_canvas_prior_fn(self, z, all_embeddings, beam_size, num_time_steps=None):

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings, num_time_steps)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size, num_time_steps)

        _, attention = self.get_canvases(z, num_time_steps)

        generate_canvas_prior = theano.function(inputs=[z],
                                                outputs=[z, x_gen_sampled, x_gen_argmax, x_gen_beam, attention],
                                                updates=updates,
                                                allow_input_downcast=True
                                                )

        return generate_canvas_prior

    def follow_latent_trajectory_fn(self, all_embeddings, alphas, num_samples, beam_size):

        z1 = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)
        z2 = self.dist_z.get_samples(dims=[1, self.z_dim], num_samples=num_samples)  # S * dim(z)

        z1_rep = T.extra_ops.repeat(z1, alphas.shape[0], axis=0)  # (S*A) * dim(z)
        z2_rep = T.extra_ops.repeat(z2, alphas.shape[0], axis=0)  # (S*A) * dim(z)

        alphas_rep = T.tile(alphas, num_samples)  # (S*A)

        z = (T.shape_padright(alphas_rep) * z1_rep) + (T.shape_padright(T.ones_like(alphas_rep) - alphas_rep) * z2_rep)
        # (S*A) * dim(z)

        x_gen_sampled, x_gen_argmax, updates = self.generate_text(z, all_embeddings)

        x_gen_beam = self.beam_search(z, all_embeddings, beam_size)

        follow_latent_trajectory = theano.function(inputs=[alphas],
                                                   outputs=[x_gen_sampled, x_gen_argmax, x_gen_beam],
                                                   updates=updates,
                                                   allow_input_downcast=True
                                                   )

        return follow_latent_trajectory

    def find_best_matches_fn(self, sentences_in, sentences_eval, sentences_eval_embedded, z, all_embeddings):

        S = sentences_in.shape[0]
        N = sentences_eval.shape[0]

        sentences_eval_rep = T.tile(sentences_eval, (S, 1))  # (S*N) * max(L)
        sentences_eval_embedded_rep = T.tile(sentences_eval_embedded, (S, 1, 1))  # (S*N) * max(L) * E

        z_rep = T.extra_ops.repeat(z, repeats=N, axis=0)  # (S*N) * dim(z)

        log_p_x, _ = self.log_p_x(sentences_eval_rep, sentences_eval_embedded_rep, sentences_eval_embedded_rep, z_rep,
                                  all_embeddings)  # (S*N)

        log_p_x = log_p_x.reshape((S, N))

        find_best_matches = theano.function(inputs=[sentences_in, sentences_eval],
                                            outputs=log_p_x,
                                            allow_input_downcast=True,
                                            on_unused_input='ignore',
                                            )

        return find_best_matches

    def get_params(self):

        canvas_rnn_params = get_all_params(self.canvas_rnn, trainable=True)

        return canvas_rnn_params + self.canvas_update_params

    def get_param_values(self):

        canvas_rnn_params_vals = get_all_param_values(self.canvas_rnn)

        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [canvas_rnn_params_vals, canvas_update_params_vals]

    def set_param_values(self, param_values):

        [canvas_rnn_params_vals, canvas_update_params_vals] = param_values

        set_all_param_values(self.canvas_rnn, canvas_rnn_params_vals)

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])


class GenAUTRWordsCustomWxx(GenAUTRWords):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        super().__init__(z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs)

        self.read_multiplier_nn = self.read_multiplier_nn_fn()

    def read_multiplier_nn_fn(self):

        l_in = InputLayer((None, self.z_dim))

        l_1 = DenseLayer(l_in, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=tanh)

        l_2 = DenseLayer(l_1, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=tanh)

        l_out = DenseLayer(l_2, num_units=self.max_length*self.embedding_dim, nonlinearity=None)

        return l_out

    def get_probs(self, x, x_dropped, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        read_multiplier = get_output(self.read_multiplier_nn, z).reshape((SN, self.max_length, self.embedding_dim))
        # (S*N) * max(L) * E

        x_read = read_multiplier * x_pre_padded  # (S*N) * max(L) * E

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))

        target_embeddings = T.dot(T.concatenate((z_rep, x_read, canvases), axis=-1),  self.canvas_update_params[-1])
        # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def beam_search(self, z, all_embeddings, beam_size):

        N = z.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, beam_size, 1))  # N * B * dim(z)

        canvases = self.get_canvases(z)[0]  # N * max(L) * E

        read_multiplier = get_output(self.read_multiplier_nn, z).reshape((N, self.max_length, self.embedding_dim))  # N
        # * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        best_edges_0 = T.zeros((N, self.vocab_size))  # N * D
        active_words_0 = -T.ones((N, beam_size))  # N * B

        def step_forward(canvases_l, read_multiplier_l, best_scores_lm1, best_edges_lm1, active_words_lm1, W_x_to_x,
                         all_embeddings, z_rep):

            active_words_lm1_embedded = self.embedder(T.cast(active_words_lm1, 'int32'), all_embeddings)  # N * B * E

            active_words_lm1_read = T.shape_padaxis(read_multiplier_l, 1) * active_words_lm1_embedded  # N * B * E

            target_embeddings = T.dot(T.concatenate((z_rep, active_words_lm1_read,
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * B * D

            probs = last_d_softmax(probs_denominators)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            best_edges_l_beam_inds = T.argmax(scores, axis=1)  # N * D

            best_edges_l = active_words_lm1[T.repeat(T.arange(N), self.vocab_size), best_edges_l_beam_inds.flatten()]
            best_edges_l = best_edges_l.reshape((N, self.vocab_size))  # N * D

            return best_scores_l, T.cast(best_edges_l, 'float32'), T.cast(active_words_l, 'float32')

        ([best_scores, best_edges, active_words], _) = theano.scan(step_forward,
                                                                   sequences=[canvases.dimshuffle((1, 0, 2)),
                                                                              read_multiplier.dimshuffle((1, 0, 2))],
                                                                   outputs_info=[best_scores_0, best_edges_0,
                                                                                 active_words_0],
                                                                   non_sequences=[self.canvas_update_params[-1],
                                                                                  all_embeddings, z_rep]
                                                                   )
        # max(L) * N * B and max(L) * N * D and max(L) * N * B

        words_L = active_words[-1, :, -1]  # N

        def step_backward(best_edges_l, words_lp1):

            words_l = best_edges_l[T.arange(N), T.cast(words_lp1, 'int32')]  # N

            return words_l

        words, _ = theano.scan(step_backward,
                               sequences=best_edges[1:][::-1],
                               outputs_info=[words_L],
                               )

        words = words[::-1]  # (max(L)-1) * N

        words = T.concatenate([words, T.shape_padleft(words_L)], axis=0).T  # N * max(L)

        return T.cast(words, 'int32')

    def get_params(self):

        canvas_rnn_params = get_all_params(self.canvas_rnn, trainable=True)

        read_multiplier_nn_params = get_all_params(self.read_multiplier_nn, trainable=True)

        return canvas_rnn_params + read_multiplier_nn_params + self.canvas_update_params

    def get_param_values(self):

        canvas_rnn_params_vals = get_all_param_values(self.canvas_rnn)

        read_multiplier_nn_params_vals = get_all_param_values(self.read_multiplier_nn)

        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [canvas_rnn_params_vals, read_multiplier_nn_params_vals, canvas_update_params_vals]

    def set_param_values(self, param_values):

        [canvas_rnn_params_vals, read_multiplier_nn_params_vals, canvas_update_params_vals] = param_values

        set_all_param_values(self.canvas_rnn, canvas_rnn_params_vals)

        set_all_param_values(self.read_multiplier_nn, read_multiplier_nn_params_vals)

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])


class GenAUTRWordsCanvasFeedback(GenAUTRWords):

    def init_canvas_update_params(self):

        W_z_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.max_length))))
        W_h_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_c_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_g_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length, self.max_length))))
        b_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length,))))

        W_z_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        W_c_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))
        b_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_z_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        W_c_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))
        b_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_z_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        b_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_z_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        W_c_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))
        b_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_h_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_h_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                     self.embedding_dim))))

        W_x_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.max_length))))
        W_x_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.embedding_dim))))

        W_Cg_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length, self.embedding_dim))))

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (2*self.embedding_dim, self.embedding_dim))))

        canvas_update_params = [W_z_a, W_h_a, W_c_a, W_g_a, b_a, W_z_i, W_h_i, W_x_i, W_c_i, b_i, W_z_f, W_h_f, W_x_f,
                                W_c_f, b_f, W_z_c, W_h_c, W_x_c, b_c, W_z_o, W_h_o, W_x_o, W_c_o, b_o, W_h_Cg, W_x_Cg,
                                W_h_Cu, W_x_Cu, W_Cg_Cu, W_x_to_x]

        return canvas_update_params

    def canvas_updater(self, z):
        """
        :param z: N * dim(z) matrix

        :return canvas: N * max(L) * E tensor
        """

        c_init = T.zeros((z.shape[0], self.nn_canvas_rnn_hid_units))  # N * H
        h_init = T.zeros((z.shape[0], self.nn_canvas_rnn_hid_units))  # N * H
        canvas_init = T.zeros((z.shape[0], self.max_length, self.embedding_dim))  # N * max(L) * E
        gate_sum_init = T.zeros((z.shape[0], self.max_length))  # N * max(L)

        def step(c_tm1, h_tm1, canvas_tm1, canvas_gate_sum_tm1, z, W_z_a, W_h_a, W_c_a, W_g_a, b_a, W_z_i, W_h_i, W_x_i,
                 W_c_i, b_i, W_z_f, W_h_f, W_x_f, W_c_f, b_f, W_z_c, W_h_c, W_x_c, b_c, W_z_o, W_h_o, W_x_o, W_c_o, b_o,
                 W_h_Cg, W_x_Cg, W_h_Cu, W_x_Cu, W_Cg_Cu):

            attn = last_d_softmax(T.dot(z, W_z_a) + T.dot(h_tm1, W_h_a) + T.dot(c_tm1, W_c_a) +
                                  T.dot(canvas_gate_sum_tm1, W_g_a) + T.shape_padleft(b_a))  # N * max(L)

            # canvas_tm1_read = T.sum(T.shape_padright(canvas_gate_sum_tm1) * canvas_tm1, axis=1)  # N * E
            canvas_tm1_read = T.sum(T.shape_padright(attn) * canvas_tm1, axis=1)  # N * E

            i_t = sigmoid(T.dot(z, W_z_i) + T.dot(h_tm1, W_h_i) + T.dot(canvas_tm1_read, W_x_i) +
                          (c_tm1 * T.shape_padleft(W_c_i)) + T.shape_padleft(b_i))  # N * H
            f_t = sigmoid(T.dot(z, W_z_f) + T.dot(h_tm1, W_h_f) + T.dot(canvas_tm1_read, W_x_f) +
                          (c_tm1 * T.shape_padleft(W_c_f)) + T.shape_padleft(b_f))  # N * H
            c_t = (f_t * c_tm1) + (i_t * tanh(T.dot(z, W_z_c) + T.dot(h_tm1, W_h_c) + T.dot(canvas_tm1_read, W_x_c) +
                                              T.shape_padleft(b_c)))  # N * H
            o_t = sigmoid(T.dot(z, W_z_o) + T.dot(h_tm1, W_h_o) + T.dot(canvas_tm1_read, W_x_o) +
                          (c_t * T.shape_padleft(W_c_o)) + T.shape_padleft(b_o))  # N * H
            h_t = o_t * self.nn_canvas_rnn_hid_nonlinearity(c_t)  # N * H

            pre_softmax_gate = T.dot(h_t, W_h_Cg) + T.dot(canvas_tm1_read, W_x_Cg)  # N * max(L)

            gate_exp = T.exp(pre_softmax_gate - pre_softmax_gate.max(axis=-1, keepdims=True))  # N * max(L)
            unnormalised_gate = gate_exp * (1. - canvas_gate_sum_tm1)  # N * max(L)
            canvas_gate = unnormalised_gate / T.sum(unnormalised_gate, axis=-1, keepdims=True)  # N * max(L)
            canvas_gate *= (1. - canvas_gate_sum_tm1)  # N * max(L)

            canvas_gate_sum = canvas_gate_sum_tm1 + canvas_gate  # N * max(L)

            canvas_update = T.dot(h_t, W_h_Cu) + T.dot(canvas_tm1_read, W_x_Cu) + T.dot(canvas_gate, W_Cg_Cu)  # N * E
            canvas_update = T.shape_padaxis(canvas_update, 1)  # N * 1 * E

            canvas_new = ((1. - T.shape_padright(canvas_gate)) * canvas_tm1) + \
                         (T.shape_padright(canvas_gate) * canvas_update)  # N * max(L) * E

            return T.cast(c_t, 'float32'), T.cast(h_t, 'float32'), T.cast(canvas_new, 'float32'), \
                   T.cast(canvas_gate_sum, 'float32')

        ([c, h, canvases, canvas_gate_sums], _) = theano.scan(step,
                                                              n_steps=self.nn_canvas_rnn_time_steps,
                                                              outputs_info=[c_init, h_init, canvas_init, gate_sum_init],
                                                              non_sequences=[z] + self.canvas_update_params[:29],
                                                              )

        return canvases[-1], canvas_gate_sums[-1]

    def get_canvases(self, z, num_time_steps=None):
        """
        :param z: N * dim(z) matrix
        :param num_time_steps: int, number of RNN time steps to use

        :return canvases: N * max(L) * E tensor
        :return canvas_gate_sums: N * max(L) matrix
        """

        canvases, canvas_gate_sums = self.canvas_updater(z)  # N * max(L) * E and N * max(L)

        return canvases, canvas_gate_sums

    def get_params(self):

        return self.canvas_update_params

    def get_param_values(self):

        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [canvas_update_params_vals]

    def set_param_values(self, param_values):

        [canvas_update_params_vals] = param_values

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])


class GenAUTRWordsAttentiveWriting(GenAUTRWords):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        super().__init__(z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs)

        self.read_attention_nn = self.read_attention_nn_fn()

    def init_canvas_update_params(self):

        W_h_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_h_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                     self.embedding_dim))))

        W_Cg_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length, self.embedding_dim))))

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim + 3*self.embedding_dim,
                                                                       self.embedding_dim))))

        canvas_update_params = [W_h_Cg, W_h_Cu, W_Cg_Cu, W_x_to_x]

        return canvas_update_params

    def read_attention_nn_fn(self):

        l_in = InputLayer((None, self.z_dim))

        l_1 = DenseLayer(l_in, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=tanh)

        l_2 = DenseLayer(l_1, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=tanh)

        l_out = DenseLayer(l_2, num_units=self.max_length**2, nonlinearity=None)

        return l_out

    def read_attention(self, z):

        N = z.shape[0]

        read_attention = get_output(self.read_attention_nn, z).reshape((N, self.max_length, self.max_length))  # N *
        # max(L) * max(L)

        read_attention_mask = T.switch(T.eq(T.tril(T.ones((self.max_length, self.max_length)), -2), 0), -np.inf, 1.)
        # max(L) * max(L)

        read_attention_pre_softmax = read_attention * T.shape_padleft(read_attention_mask)  # N * max(L) * max(L)

        read_attention_pre_softmax = T.switch(T.isinf(read_attention_pre_softmax), -np.inf, read_attention_pre_softmax)
        # N * max(L) * max(L)

        read_attention_softmax = last_d_softmax(read_attention_pre_softmax)  # N * max(L) * max(L)

        read_attention_softmax = T.switch(T.isnan(read_attention_softmax), 0,
                                          read_attention_softmax)  # N * max(L) * max(L)

        return read_attention_softmax

    def get_probs(self, x, x_dropped, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param x_dropped: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x_dropped], axis=1)[:, :-1]  # (S*N) *
        # max(L) * E

        read_attention = self.read_attention(z)  # (S*N) * max(L) * max(L)

        total_written = T.batched_dot(read_attention, x_dropped)  # (S*N) * max(L) * E

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))

        target_embeddings = T.dot(T.concatenate((z_rep, total_written, x_pre_padded, canvases), axis=-1),
                                  self.canvas_update_params[-1])  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def beam_search(self, z, all_embeddings, beam_size, num_time_steps=None):

        N = z.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, beam_size, 1))  # N * B * dim(z)

        canvases = self.get_canvases(z, num_time_steps)[0]  # N * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z)  # N * max(L) * max(L)

        def step_forward(l, best_scores_lm1, active_paths_current, canvases, all_embeddings, W_x_to_x, z_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((N*beam_size, self.max_length,
                                                                                 self.embedding_dim))
                                          ).reshape((N, beam_size, self.max_length, self.embedding_dim))[:, :, l]  # N *
            # B * E

            target_embeddings = T.dot(T.concatenate((z_rep, total_written, active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * B * D

            probs = last_d_softmax(probs_denominators)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()].reshape(
                (N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return best_scores_l, active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=T.arange(self.max_length),
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[canvases, all_embeddings,
                                                                      self.canvas_update_params[-1], z_rep]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def impute_missing_words(self, z, x_best_guess, missing_words_mask, all_embeddings, beam_size):

        N = z.shape[0]

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, beam_size, 1))  # N * B * dim(z)

        canvases = self.get_canvases(z)[0]  # N * max(L) * E

        best_scores_0 = T.zeros((N, beam_size))  # N * B
        active_paths_init = -T.ones((N, beam_size, self.max_length))  # N * B * max(L)

        read_attention = self.read_attention(z)  # N * max(L) * max(L)

        def step_forward(l, x_best_guess_l, missing_words_mask_l, best_scores_lm1, active_paths_current, canvases,
                         all_embeddings, W_x_to_x, z_rep):

            canvases_l = canvases[:, l]

            active_paths_current_embedded = self.embedder(T.cast(active_paths_current, 'int32'), all_embeddings)  # N *
            # B * max(L) * E

            total_written = T.batched_dot(T.tile(read_attention, (beam_size, 1, 1)),
                                          active_paths_current_embedded.reshape((N*beam_size, self.max_length,
                                                                                 self.embedding_dim))
                                          ).reshape((N, beam_size, self.max_length, self.embedding_dim))[:, :, l]  # N *
            # B * E

            target_embeddings = T.dot(T.concatenate((z_rep, total_written, active_paths_current_embedded[:, :, l-1],
                                                     T.tile(T.shape_padaxis(canvases_l, 1), (1, beam_size, 1))),
                                                    axis=-1),
                                      W_x_to_x)  # N * B * E

            probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # N * B * D

            probs = last_d_softmax(probs_denominators)  # N * B * D

            scores = T.shape_padright(best_scores_lm1) + T.log(probs)  # N * B * D

            best_scores_l_all = T.max(scores, axis=1)  # N * D

            best_scores_l = T.sort(best_scores_l_all, axis=-1)[:, -beam_size:]  # N * B

            best_scores_l_known = best_scores_l_all[T.arange(N), x_best_guess_l]  # N

            best_scores_l = T.switch(T.eq(T.shape_padright(missing_words_mask_l), 0.), best_scores_l,
                                     T.shape_padright(best_scores_l_known))  # N * B

            active_words_l = T.argsort(best_scores_l_all, axis=1)[:, -beam_size:]  # N * B

            active_words_l = T.switch(T.eq(T.shape_padright(missing_words_mask_l), 0.), active_words_l,
                                      T.shape_padright(x_best_guess_l))  # N * B

            best_paths_l_all = T.argmax(scores, axis=1)  # N * D

            best_paths_l_inds = best_paths_l_all[T.repeat(T.arange(N), beam_size), active_words_l.flatten()]  # (N*B)
            best_paths_l_inds = best_paths_l_inds.reshape((N, beam_size))  # N * B

            best_paths_l = active_paths_current[T.repeat(T.arange(N), beam_size), best_paths_l_inds.flatten()]  # (N*B)
            # * max(L)
            best_paths_l = best_paths_l.reshape((N, beam_size, self.max_length))  # N * B * max(L)

            active_paths_new = T.set_subtensor(best_paths_l[:, :, l], active_words_l)

            return best_scores_l, active_paths_new

        ([best_scores, active_paths], _) = theano.scan(step_forward,
                                                       sequences=[T.arange(self.max_length), x_best_guess.T,
                                                                  missing_words_mask.T],
                                                       outputs_info=[best_scores_0, active_paths_init],
                                                       non_sequences=[canvases, all_embeddings,
                                                                      self.canvas_update_params[-1], z_rep]
                                                       )
        # max(L) * N * B and max(L) * N * B * max(L)

        active_paths = active_paths[-1]  # N * B * max(L)

        words = active_paths[:, -1]  # N * max(L)

        return T.cast(words, 'int32')

    def get_params(self):

        canvas_rnn_params = get_all_params(self.canvas_rnn, trainable=True)

        read_attention_nn_params = get_all_params(self.read_attention_nn, trainable=True)

        return canvas_rnn_params + read_attention_nn_params + self.canvas_update_params

    def get_param_values(self):

        canvas_rnn_params_vals = get_all_param_values(self.canvas_rnn)

        read_attention_nn_params_vals = get_all_param_values(self.read_attention_nn)

        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [canvas_rnn_params_vals, read_attention_nn_params_vals, canvas_update_params_vals]

    def set_param_values(self, param_values):

        [canvas_rnn_params_vals, read_attention_nn_params_vals, canvas_update_params_vals] = param_values

        set_all_param_values(self.canvas_rnn, canvas_rnn_params_vals)

        set_all_param_values(self.read_attention_nn, read_attention_nn_params_vals)

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])


class GenAUTRWordsCanvasFeedbackAttentiveWriting(GenAUTRWordsCanvasFeedback):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        super().__init__(z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs)

        self.read_attention_nn = self.read_attention_nn_fn()

    def init_canvas_update_params(self):

        W_z_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.max_length))))
        W_h_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_c_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_g_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length, self.max_length))))
        b_a = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length,))))

        W_z_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        W_c_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))
        b_i = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_z_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        W_c_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))
        b_f = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_z_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        b_c = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_z_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim, self.nn_canvas_rnn_hid_units))))
        W_h_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                    self.nn_canvas_rnn_hid_units))))
        W_x_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.nn_canvas_rnn_hid_units))))
        W_c_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))
        b_o = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,))))

        W_h_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units, self.max_length))))
        W_h_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.nn_canvas_rnn_hid_units,
                                                                     self.embedding_dim))))

        W_x_Cg = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.max_length))))
        W_x_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.embedding_dim, self.embedding_dim))))

        W_Cg_Cu = theano.shared(np.float32(np.random.normal(0., 0.1, (self.max_length, self.embedding_dim))))

        W_x_to_x = theano.shared(np.float32(np.random.normal(0., 0.1, (self.z_dim + 3*self.embedding_dim,
                                                                       self.embedding_dim))))

        canvas_update_params = [W_z_a, W_h_a, W_c_a, W_g_a, b_a, W_z_i, W_h_i, W_x_i, W_c_i, b_i, W_z_f, W_h_f, W_x_f,
                                W_c_f, b_f, W_z_c, W_h_c, W_x_c, b_c, W_z_o, W_h_o, W_x_o, W_c_o, b_o, W_h_Cg, W_x_Cg,
                                W_h_Cu, W_x_Cu, W_Cg_Cu, W_x_to_x]

        return canvas_update_params

    def read_attention_nn_fn(self):

        l_in = InputLayer((None, self.z_dim))

        l_1 = DenseLayer(l_in, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=tanh)

        l_2 = DenseLayer(l_1, num_units=self.nn_canvas_rnn_hid_units, nonlinearity=tanh)

        l_out = DenseLayer(l_2, num_units=self.max_length**2, nonlinearity=None)

        return l_out

    def get_probs(self, x, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param z: (S*N) * dim(z) matrix
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x], axis=1)[:, :-1]  # (S*N) * max(L) * E

        read_attention = get_output(self.read_attention_nn, z).reshape((SN, self.max_length, self.max_length))  # (S*N)
        # * max(L) * max(L)

        read_attention_mask = T.switch(T.eq(T.tril(T.ones((self.max_length, self.max_length)), -2), 0), -np.inf, 1.)
        # max(L) * max(L)

        read_attention_pre_softmax = read_attention * T.shape_padleft(read_attention_mask)  # (S*N) * max(L) * max(L)

        read_attention_pre_softmax = T.switch(T.isinf(read_attention_pre_softmax), -np.inf, read_attention_pre_softmax)
        # (S*N) * max(L) * max(L)

        read_attention_softmax = last_d_softmax(read_attention_pre_softmax)  # (S*N) * max(L) * max(L)

        read_attention_softmax = T.switch(T.isnan(read_attention_softmax), 0, read_attention_softmax).dimshuffle((0, 2, 1))
        # (S*N) * max(L) * max(L)

        total_written = T.batched_dot(x.dimshuffle((0, 2, 1)), read_attention_softmax).dimshuffle((0, 2, 1))  # (S*N) *
        # max(L) * E

        z_rep = T.tile(T.shape_padaxis(z, 1), (1, self.max_length, 1))

        target_embeddings = T.dot(T.concatenate((z_rep, total_written, x_pre_padded, canvases), axis=-1),
                                  self.canvas_update_params[-1])  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def get_params(self):

        read_attention_nn_params = get_all_params(self.read_attention_nn, trainable=True)

        return read_attention_nn_params + self.canvas_update_params

    def get_param_values(self):

        read_attention_nn_params_vals = get_all_param_values(self.read_attention_nn)

        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [read_attention_nn_params_vals, canvas_update_params_vals]

    def set_param_values(self, param_values):

        [read_attention_nn_params_vals, canvas_update_params_vals] = param_values

        set_all_param_values(self.read_attention_nn, read_attention_nn_params_vals)

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])


class GenAUTRWordsRNN(GenAUTRWords):

    def __init__(self, z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs):

        super().__init__(z_dim, max_length, vocab_size, embedding_dim, embedder, dist_z, dist_x, nn_kwargs)

        self.text_gen_rnn_input_layers, self.text_gen_rnn_output_layer = self.text_gen_rnn_fn()

    def text_gen_rnn_fn(self):

        l_in_z = InputLayer((None, None, self.z_dim))
        l_in_x = InputLayer((None, None, self.embedding_dim))
        l_in_canvas = InputLayer((None, None, self.embedding_dim))

        l_in = ConcatLayer([l_in_z, l_in_x, l_in_canvas], axis=-1)

        l_hid = LSTMLayer(l_in, num_units=self.nn_canvas_rnn_hid_units,
                          nonlinearity=self.nn_canvas_rnn_hid_nonlinearity)

        l_out = RecurrentLayer(l_hid, num_units=self.embedding_dim, W_hid_to_hid=T.zeros, nonlinearity=None)

        return (l_in_z, l_in_x, l_in_canvas), l_out

    def get_probs(self, x, z, canvases, all_embeddings, mode='all'):
        """
        :param x: (S*N) * max(L) * E tensor
        :param canvases: (S*N) * max(L) * E matrix
        :param all_embeddings: D * E matrix
        :param mode: 'all' returns probabilities for every element in the vocabulary, 'true' returns only the
        probability for the true word.

        :return probs: (S*N) * max(L) * D tensor or (S*N) * max(L) matrix
        """

        SN = x.shape[0]

        z_rep = T.tile(z.reshape((z.shape[0], 1, z.shape[1])), (1, self.max_length, 1))

        x_pre_padded = T.concatenate([T.zeros((SN, 1, self.embedding_dim)), x], axis=1)[:, :-1]  # (S*N) * max(L) * E

        target_embeddings = get_output(self.text_gen_rnn_output_layer,
                                       {self.text_gen_rnn_input_layers[0]: z_rep,
                                        self.text_gen_rnn_input_layers[1]: x_pre_padded,
                                        self.text_gen_rnn_input_layers[2]: canvases})  # (S*N) * max(L) * E

        probs_numerators = T.sum(x * target_embeddings, axis=-1)  # (S*N) * max(L)

        probs_denominators = T.dot(target_embeddings, all_embeddings.T)  # (S*N) * max(L) * D

        if mode == 'all':
            probs = last_d_softmax(probs_denominators)  # (S*N) * max(L) * D
        elif mode == 'true':
            probs_numerators -= T.max(probs_denominators, axis=-1)
            probs_denominators -= T.max(probs_denominators, axis=-1, keepdims=True)

            probs = T.exp(probs_numerators) / T.sum(T.exp(probs_denominators), axis=-1)  # (S*N) * max(L)
        else:
            raise Exception("mode must be in ['all', 'true']")

        return probs

    def get_params(self):

        canvas_rnn_params = get_all_params(self.canvas_rnn, trainable=True)

        return canvas_rnn_params + self.canvas_update_params

    def get_param_values(self):

        canvas_rnn_params_vals = get_all_param_values(self.canvas_rnn)

        canvas_update_params_vals = [p.get_value() for p in self.canvas_update_params]

        return [canvas_rnn_params_vals, canvas_update_params_vals]

    def set_param_values(self, param_values):

        [canvas_rnn_params_vals, canvas_update_params_vals] = param_values

        set_all_param_values(self.canvas_rnn, canvas_rnn_params_vals)

        for i in range(len(self.canvas_update_params)):
            self.canvas_update_params[i].set_value(canvas_update_params_vals[i])
