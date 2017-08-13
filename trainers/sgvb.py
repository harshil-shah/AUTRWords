import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import norm_constraint


class SGVBWords(object):

    def __init__(self, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim, dist_z_gen,
                 dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs):

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

    def init_generative_model(self, generative_model):

        return generative_model(self.z_dim, self.max_length, self.vocab_size, self.embedding_dim, self.embedder,
                                self.dist_z_gen, self.dist_x_gen, self.gen_nn_kwargs)

    def init_recognition_model(self, recognition_model):

        return recognition_model(self.z_dim, self.max_length, self.embedding_dim, self.dist_z_rec, self.rec_nn_kwargs)

    def embedder(self, x, all_embeddings):

        all_embeddings = T.concatenate([all_embeddings, T.zeros((1, self.embedding_dim))], axis=0)

        return all_embeddings[x]

    def symbolic_elbo(self, x, num_samples, beta=None):

        x_embedded = self.embedder(x, self.all_embeddings)  # N * max(L) * E

        z = self.recognition_model.get_samples(x, x_embedded, num_samples)  # (S*N) * dim(z)

        log_p_x, pp = self.generative_model.log_p_x(x, x_embedded, z, self.all_embeddings)  # (S*N)

        kl = self.recognition_model.kl_std_gaussian(x, x_embedded)  # N

        if beta is None:
            elbo = T.sum(((1. / num_samples) * log_p_x) - kl)
        else:
            elbo = T.sum(((1. / num_samples) * log_p_x) - (beta * kl))

        return elbo, T.sum(kl), T.sum((1./num_samples) * pp)

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

        elbo, kl, pp = self.symbolic_elbo(x, num_samples, beta)

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

        optimiser = theano.function(inputs=[x, beta],
                                    outputs=[elbo, kl, pp],
                                    updates=updates,
                                    allow_input_downcast=True,
                                    on_unused_input='ignore',
                                    )

        return optimiser, updates

    def generate_output_prior_fn(self, num_samples, beam_size):

        return self.generative_model.generate_output_prior_fn(self.all_embeddings, num_samples, beam_size)

    def generate_output_posterior_fn(self, beam_size):

        x = T.imatrix('x')  # N * max(L)

        x_embedded = self.embedder(x, self.all_embeddings)

        z = self.recognition_model.get_samples(x, x_embedded, 1, means_only=True)  # N * dim(z) matrix

        return self.generative_model.generate_output_posterior_fn(x, z, self.all_embeddings, beam_size)


class SGVBStanfordWords(SGVBWords):

    def __init__(self, generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim, dist_z_gen,
                 dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs):

        super().__init__(generative_model, recognition_model, z_dim, max_length, vocab_size, embedding_dim, dist_z_gen,
                         dist_x_gen, dist_z_rec, gen_nn_kwargs, rec_nn_kwargs)

    def symbolic_elbo(self, x, num_samples, beta=None, drop_mask=None):

        x_embedded = self.embedder(x, self.all_embeddings)  # N * max(L) * E

        z = self.recognition_model.get_samples(x, x_embedded, num_samples)  # (S*N) * dim(z)

        if drop_mask is None:
            x_embedded_dropped = x_embedded
        else:
            x_embedded_dropped = x_embedded * T.shape_padright(drop_mask)

        log_p_x, pp = self.generative_model.log_p_x(x, x_embedded, x_embedded_dropped, z, self.all_embeddings)  # (S*N)

        kl = self.recognition_model.kl_std_gaussian(x, x_embedded)  # N

        if beta is None:
            elbo = T.sum(((1. / num_samples) * log_p_x) - kl)
        else:
            elbo = T.sum(((1. / num_samples) * log_p_x) - (beta * kl))

        return elbo, T.sum(kl), T.sum((1./num_samples) * pp)

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
