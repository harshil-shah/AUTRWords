import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import norm_constraint


class SGVBWords(object):

    def __init__(self, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim, dist_z_gen,
                 dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs, eos_ind):

        self.z_dim = z_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.all_embeddings = theano.shared(np.float32(np.random.normal(0., 0.1, (vocab_size, embedding_dim))))

        self.dist_z_gen = dist_z_gen
        self.dist_x_gen = dist_x_gen
        self.dist_z_rec = dist_z_rec

        self.gen_nn_kwargs = gen_nn_kwargs
        self.rec_nn_kwargs = rec_nn_kwargs

        self.generative_model = self.init_generative_model(generative_model)
        self.recognition_model = self.init_recognition_model(recognition_model)

        self.eos_ind = eos_ind

    def init_generative_model(self, generative_model):

        return generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim, self.embedder,
                                self.dist_z_gen, self.dist_x_gen, self.gen_nn_kwargs)

    def init_recognition_model(self, recognition_model):

        return recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec, self.rec_nn_kwargs)

    def embedder(self, x, all_embeddings):

        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, self.embedding_dim))], axis=0)

        return all_embeddings[x]

    def cut_off(self, x):

        def step(x_l, x_lm1):

            x_l = T.switch(T.eq(x_lm1, self.eos_ind), -1, x_l)
            x_l = T.switch(T.eq(x_lm1, -1), -1, x_l)

            return T.cast(x_l, 'int32')

        x_cut_off, _ = theano.scan(step,
                                   sequences=x.T,
                                   outputs_info=T.zeros((x.shape[0],), 'int32'),
                                   )

        return x_cut_off.T

    def symbolic_elbo(self, x, num_samples, beta=None, drop_mask=None):

        x_embedded = self.embedder(x, self.all_embeddings)  # N * max(L) * E

        z, kl = self.recognition_model.get_samples_and_kl_std_gaussian(x, x_embedded, num_samples)  # (S*N) * dim(z) and
        # N

        if drop_mask is None:
            x_embedded_dropped = x_embedded
        else:
            x_embedded_dropped = x_embedded * T.shape_padright(drop_mask)

        log_p_x = self.generative_model.log_p_x(x, x_embedded, x_embedded_dropped, z, self.all_embeddings)  # (S*N)

        if beta is None:
            elbo = T.sum((1. / num_samples) * log_p_x) - T.sum(kl)
        else:
            elbo = T.sum((1. / num_samples) * log_p_x) - T.sum(beta * kl)

        pp = T.exp(-(T.sum((1. / num_samples) * log_p_x) - T.sum(kl)) / T.sum(T.switch(T.lt(x, 0), 0, 1)))

        return elbo, T.sum(kl), pp

    def elbo_fn(self, num_samples):

        x = T.imatrix('x')  # N * max(L)

        elbo, kl, pp = self.symbolic_elbo(x, num_samples)

        elbo_fn = theano.function(inputs=[x],
                                  outputs=[elbo, kl, pp],
                                  allow_input_downcast=True,
                                  )

        return elbo_fn

    def optimiser(self, num_samples, grad_norm_constraint, update, update_kwargs, saved_update=None):

        x = T.imatrix('x')  # N * max(L)

        beta = T.scalar('beta')

        drop_mask = T.matrix('drop_mask')  # N * max(L)

        elbo, kl, pp = self.symbolic_elbo(x, num_samples, beta, drop_mask)

        params = self.generative_model.get_params() + self.recognition_model.get_params() + [self.all_embeddings]
        grads = T.grad(-elbo, params, disconnected_inputs='ignore')

        if grad_norm_constraint is not None:
            grads = [norm_constraint(g, grad_norm_constraint) if g.ndim > 1 else g for g in grads]

        update_kwargs['loss_or_grads'] = grads
        update_kwargs['params'] = params

        updates = update(**update_kwargs)

        if saved_update is not None:
            for u, v in zip(updates, saved_update.keys()):
                u.set_value(v.get_value())

        optimiser = theano.function(inputs=[x, beta, drop_mask],
                                    outputs=[elbo, kl, pp],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples, beam_size, num_time_steps=None):

        outputs, updates = self.generative_model.generate_output_prior(self.all_embeddings, num_samples, beam_size,
                                                                       num_time_steps)

        return theano.function(inputs=[],
                               outputs=outputs,
                               updates=updates,
                               allow_input_downcast=True,
                               )

    def generate_output_posterior_fn(self, beam_size, num_time_steps=None):

        x = T.imatrix('x')  # N * max(L)

        x_embedded = self.embedder(x, self.all_embeddings)

        z = self.recognition_model.get_samples(x, x_embedded, 1, means_only=True)  # N * dim(z) matrix

        return self.generative_model.generate_output_posterior_fn(x, z, self.all_embeddings, beam_size, num_time_steps)

    def generate_canvases_prior_fn(self, num_samples, beam_size):

        time_steps = self.generative_model.nn_canvas_rnn_time_steps

        z = T.matrix('z')  # S * dim(z)

        fns = []

        for t in range(time_steps):

            fns.append(self.generative_model.generate_canvas_prior_fn(z, self.all_embeddings, beam_size, t+1))

        return fns

    def impute_missing_words_fn(self, beam_size):

        x_best_guess = T.imatrix('x_best_guess')  # N * max(L)

        missing_words_mask = T.matrix('drop_mask')  # N * max(L)

        x_best_guess_embedded = self.embedder(x_best_guess, self.all_embeddings)  # N * max(L) * E

        z = self.recognition_model.get_samples(x_best_guess, x_best_guess_embedded, 1, means_only=True)  # N *
        # dim(z)

        x_best_guess_new = self.generative_model.impute_missing_words(z, x_best_guess, missing_words_mask,
                                                                      self.all_embeddings, beam_size)  # N * max(L)

        return theano.function(inputs=[x_best_guess, missing_words_mask],
                               outputs=x_best_guess_new,
                               allow_input_downcast=True,
                               )

    def follow_latent_trajectory_fn(self, num_samples, beam_size):

        alphas = T.vector('alphas')

        return self.generative_model.follow_latent_trajectory_fn(self.all_embeddings, alphas, num_samples, beam_size)

    def find_best_matches_fn(self):

        sentences_in = T.imatrix('sentences_in')  # S * max(L) matrix
        sentences_eval = T.imatrix('sentences_eval')  # N * max(L) matrix

        sentences_in_embedded = self.embedder(sentences_in, self.all_embeddings)  # S * max(L) * E
        sentences_eval_embedded = self.embedder(sentences_eval, self.all_embeddings)  # N * max(L) * E

        z = self.recognition_model.get_samples(sentences_in, sentences_in_embedded, 1, means_only=True)  # S * dim(z)

        return self.generative_model.find_best_matches_fn(sentences_in, sentences_eval, sentences_eval_embedded, z,
                                                          self.all_embeddings)
