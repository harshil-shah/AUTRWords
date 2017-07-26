from collections import OrderedDict
import os
import pickle as cPickle
import time
import numpy as np
import json
from lasagne.updates import adam
from data_processing.utilities import chunker


np.random.seed(1234)


class RunWords(object):

    def __init__(self, solver, solver_kwargs, valid_vocab, main_dir, out_dir, dataset, load_param_dir=None,
                 pre_trained=False, restrict_min_length=None, restrict_max_length=None,
                 train_prop=0.95):

        self.valid_vocab = valid_vocab

        self.main_dir = main_dir
        self.out_dir = out_dir
        self.load_param_dir = load_param_dir

        self.solver_kwargs = solver_kwargs

        self.vocab_size = solver_kwargs['vocab_size']

        self.X_train, self.X_test, self.L_train, self.L_test = self.load_data(dataset, train_prop, restrict_min_length,
                                                                              restrict_max_length)

        print('# training sentences = ' + str(len(self.L_train)))
        print('# test sentences = ' + str(len(self.L_test)))

        self.max_length = np.concatenate((self.X_train, self.X_test), axis=0).shape[1]

        self.vb = solver(max_length=self.max_length, **self.solver_kwargs)

        self.pre_trained = pre_trained

        if self.pre_trained:

            with open(os.path.join(self.load_param_dir, 'gen_params.save'), 'rb') as f:
                self.vb.generative_model.set_param_values(cPickle.load(f))

            with open(os.path.join(self.load_param_dir, 'recog_params.save'), 'rb') as f:
                self.vb.recognition_model.set_param_values(cPickle.load(f))

    def load_data(self, dataset, train_prop, restrict_min_length, restrict_max_length):

        folder = '../_datasets/' + dataset

        files = []

        for f in os.listdir(folder):

            try:
                lower = int(f[:f.find('-')])
                upper = int(f[f.find('-')+1:f.find('.')])
            except:
                continue

            if lower > restrict_max_length or upper < restrict_min_length:
                continue
            else:
                files.append(f)

        words = []

        for f in files:

            with open(os.path.join(self.main_dir, folder, f), 'r') as d:
                words_d = d.read()
                words += json.loads(words_d)

        L = np.array([len(s) for s in words])

        mask = L[:, None] > np.arange(max(L))

        words_to_return = np.full(mask.shape, -1, dtype='int')
        words_to_return[mask] = np.concatenate(words)

        training_mask = np.random.rand(len(words_to_return)) < train_prop

        return words_to_return[training_mask], words_to_return[~training_mask], L[training_mask], L[~training_mask]

    def call_elbo_fn(self, elbo_fn, x):

        return elbo_fn(x)

    def call_optimiser(self, optimiser, x, beta):

        return optimiser(x, beta)

    def get_generate_output_prior(self, num_outputs):

        return self.vb.generate_output_prior_fn(num_outputs)

    def call_generate_output_prior(self, generate_output_prior):

        z, x_gen_sampled, x_gen_argmax = generate_output_prior()

        out = OrderedDict()

        out['generated_z_prior'] = z
        out['generated_x_sampled_prior'] = x_gen_sampled
        out['generated_x_argmax_prior'] = x_gen_argmax

        return out

    def print_output_prior(self, output_prior):

        x_gen_sampled = output_prior['generated_x_sampled_prior']
        x_gen_argmax = output_prior['generated_x_argmax_prior']

        print('='*10)

        for n in range(x_gen_sampled.shape[0]):

            print('gen x sampled: ' + ' '.join([self.valid_vocab[int(i)] for i in x_gen_sampled[n]]))
            print(' gen x argmax: ' + ' '.join([self.valid_vocab[int(i)] for i in x_gen_argmax[n]]))

            print('-'*10)

        print('='*10)

    def get_generate_output_posterior(self):

        return self.vb.generate_output_posterior_fn()

    def call_generate_output_posterior(self, generate_output_posterior, x):

        z, x_gen_sampled, x_gen_argmax = generate_output_posterior(x)

        out = OrderedDict()

        out['true_x_for_posterior'] = x
        out['generated_z_posterior'] = z
        out['generated_x_sampled_posterior'] = x_gen_sampled
        out['generated_x_argmax_posterior'] = x_gen_argmax

        return out

    def print_output_posterior(self, output_posterior):

        x = output_posterior['true_x_for_posterior']
        x_gen_sampled = output_posterior['generated_x_sampled_posterior']
        x_gen_argmax = output_posterior['generated_x_argmax_posterior']

        valid_vocab_for_true = self.valid_vocab + ['']

        print('='*10)

        for n in range(x.shape[0]):

            print('       true x: ' + ' '.join([valid_vocab_for_true[i] for i in x[n]]).strip())
            print('gen x sampled: ' + ' '.join([self.valid_vocab[int(i)] for i in x_gen_sampled[n]]))
            print(' gen x argmax: ' + ' '.join([self.valid_vocab[int(i)] for i in x_gen_argmax[n]]))

            print('-'*10)

        print('='*10)

    def train(self, n_iter, batch_size, num_samples, grad_norm_constraint=None, update=adam, update_kwargs=None,
              warm_up=None, val_freq=None, val_batch_size=0, val_num_samples=0,
              val_print_gen=5, save_params_every=None):

        if self.pre_trained:
            with open(os.path.join(self.load_param_dir, 'updates.save'), 'rb') as f:
                saved_update = cPickle.load(f)
        else:
            saved_update = None

        optimiser, updates = self.vb.optimiser(num_samples=num_samples, grad_norm_constraint=grad_norm_constraint,
                                               update=update, update_kwargs=update_kwargs, saved_update=saved_update)

        elbo_fn = self.vb.elbo_fn(val_num_samples)

        generate_output_prior = self.get_generate_output_prior(val_print_gen)
        generate_output_posterior = self.get_generate_output_posterior()

        for i in range(n_iter):

            start = time.clock()

            batch_indices = np.random.choice(len(self.X_train), batch_size)
            batch = np.array([self.X_train[ind] for ind in batch_indices])

            beta = 1. if warm_up is None or i > warm_up else float(i) / warm_up

            elbo, kl = self.call_optimiser(optimiser, batch, beta)

            print('Iteration ' + str(i + 1) + ': ELBO = ' + str(elbo/batch_size) + ' (KL = ' + str(kl/batch_size) +
                  ') per data point (time taken = ' + str(time.clock() - start) + ' seconds)')

            if val_freq is not None and i % val_freq == 0:

                val_batch_indices = np.random.choice(len(self.X_test), val_batch_size, replace=False)
                val_batch = np.array([self.X_test[ind] for ind in val_batch_indices])

                val_elbo, val_kl = self.call_elbo_fn(elbo_fn, val_batch)

                print('Test set ELBO = ' + str(val_elbo/val_batch_size) + ' (KL = ' + str(kl/batch_size) +
                      ') per data point')

                output_prior = self.call_generate_output_prior(generate_output_prior)

                self.print_output_prior(output_prior)

                post_batch_indices = np.random.choice(len(self.X_train), val_print_gen, replace=False)
                post_batch_in = np.array([self.X_train[ind] for ind in post_batch_indices])

                output_posterior = self.call_generate_output_posterior(generate_output_posterior, post_batch_in)

                self.print_output_posterior(output_posterior)

            if save_params_every is not None and i % save_params_every == 0 and i > 0:

                with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
                    cPickle.dump(self.vb.generative_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
                    cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

                with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
                    cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'gen_params.save'), 'wb') as f:
            cPickle.dump(self.vb.generative_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'recog_params.save'), 'wb') as f:
            cPickle.dump(self.vb.recognition_model.get_param_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, 'updates.save'), 'wb') as f:
            cPickle.dump(updates, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def test(self, batch_size, num_samples, sub_sample_size=None):

        elbo_fn = self.vb.elbo_fn(num_samples) if sub_sample_size is None else self.vb.elbo_fn(sub_sample_size)

        elbo = 0
        kl = 0

        batches_complete = 0

        for batch_X in chunker([self.X_test], batch_size):

            start = time.clock()

            if sub_sample_size is None:

                elbo_batch, kl_batch = self.call_elbo_fn(elbo_fn, batch_X)

            else:

                elbo_batch = 0
                kl_batch = 0

                for sub_sample in range(1, (num_samples / sub_sample_size) + 1):

                    elbo_sub_batch, kl_sub_batch = self.call_elbo_fn(elbo_fn, batch_X)

                    elbo_batch = (elbo_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
                                                float(sub_sample * sub_sample_size))) + \
                                 (elbo_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))

                    kl_batch = (kl_batch * (float((sub_sample * sub_sample_size) - sub_sample_size) /
                                            float(sub_sample * sub_sample_size))) + \
                               (kl_sub_batch * float(sub_sample_size / float(sub_sample * sub_sample_size)))

            elbo += elbo_batch
            kl += kl_batch

            batches_complete += 1

            print('Tested batches ' + str(batches_complete) + ' of ' + str(round(self.X_test.shape[0] / batch_size))
                  + '; test set ELBO so far = ' + str(elbo) + ' (' + str(kl) + ')'
                  + ' / ' + str(elbo / (batches_complete * batch_size)) + ' ('
                  + str(kl / (batches_complete * batch_size)) + ') per obs.'
                  + ' (time taken = ' + str(time.clock() - start) + ' seconds)')

        print('Test set ELBO = ' + str(elbo))

    def generate_output(self, prior, posterior, num_outputs):

        if prior:

            generate_output_prior = self.vb.generate_output_prior_fn(num_outputs, only_final=False)

            output_prior = self.call_generate_output_prior(generate_output_prior)

            for key, value in output_prior.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

        if posterior:

            generate_output_posterior = self.vb.generate_output_posterior_fn(only_final=False)

            batch_indices = np.random.choice(len(self.X_train), num_outputs, replace=False)
            batch_in = np.array([self.X_train[ind] for ind in batch_indices])

            output_posterior = self.call_generate_output_posterior(generate_output_posterior, batch_in)

            for key, value in output_posterior.items():
                np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def impute_missing_chars(self, num_outputs, drop_rate, num_iterations):

        impute_missing_chars = self.vb.impute_missing_chars()

        batch_indices = np.random.choice(len(self.X_train), num_outputs, replace=False)
        batch_in = np.array([self.X_train[ind] for ind in batch_indices])
        L_in = [self.L_train[ind] for ind in batch_indices]

        missing_chars = [np.random.choice(l, size=int(round(drop_rate*l)), replace=False) for l in L_in]
        missing_chars_mask = [[0 if l in array else 1 for l in range(self.max_length)] for array in missing_chars]
        missing_chars_mask = np.array(missing_chars_mask)

        best_guess_rand = np.random.rand(num_outputs, self.max_length, self.solver_kwargs['vocab_size'])
        best_guess = np.equal(best_guess_rand, np.amax(best_guess_rand, axis=-1, keepdims=True))

        char_index_gen = self.valid_vocab
        char_index_true = ' ' + char_index_gen

        for i in range(num_iterations):

            start = time.clock()

            best_guess = impute_missing_chars(batch_in, missing_chars_mask, best_guess)

            print('Iteration ' + str(i+1) + ' (time taken = ' + str(time.clock() - start) + ' seconds)')
            print(' ')
            ind_to_print = np.random.randint(num_outputs)
            print(''.join([char_index_true[i] for i in batch_in[ind_to_print]]))
            print(''.join([char_index_true[int(batch_in[ind_to_print][i])] if i not in missing_chars[ind_to_print]
                           else '_' for i in range(len(batch_in[ind_to_print]))]))
            print(''.join([char_index_gen[int(np.argmax(i))] for i in best_guess[ind_to_print]]))
            print(' ')

        num_missing_chars = len([item for sublist in missing_chars for item in sublist])

        best_guess_chars = np.argmax(best_guess, axis=2)

        num_correct_chars = 0

        for n in range(num_outputs):
            num_correct_chars_n = sum([1 if batch_in[n, l] == best_guess_chars[n, l] else 0 for l in missing_chars[n]])
            num_correct_chars += num_correct_chars_n

        prop_correct_chars = float(num_correct_chars) / num_missing_chars

        print('proportion correct characters = ' + str(prop_correct_chars))

        out = OrderedDict()

        out['true_X_for_misschar'] = batch_in
        out['misschar_missing_chars'] = missing_chars
        out['misschar_X'] = best_guess

        for key, value in out.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def find_best_matches(self, num_outputs, num_matches, batch_size):

        sentences_indices = np.random.choice(len(self.X_test), num_outputs, replace=False)
        sentences = np.array([self.X_test[ind] for ind in sentences_indices])

        find_best_matches_fn = self.vb.find_best_matches_fn()

        log_probs = []

        batches_evaluated = 0

        for min_ind in range(0, len(self.X_test), batch_size):

            start = time.clock()

            batch = self.X_test[min_ind: min_ind+batch_size]

            log_probs.append(find_best_matches_fn(sentences, batch))

            batches_evaluated += 1

            print('batches evaluated = ' + str(batches_evaluated) + ' (time taken = ' + str(time.clock() - start)
                  + ' seconds)')

        log_probs = np.concatenate(log_probs, axis=1)

        log_probs /= self.L_test

        best_match_indices = np.argsort(log_probs, axis=1)[:, -num_matches:]

        best_matches = np.array([self.X_test[inds] for inds in best_match_indices])

        out = OrderedDict()

        out['best_matches_input'] = sentences
        out['best_matches_log_probs_normed'] = log_probs
        out['best_matches_output_normed'] = best_matches

        for key, value in out.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)

    def follow_latent_trajectory(self, num_samples, num_steps):

        follow_latent_trajectory = self.vb.follow_latent_trajectory_fn(num_samples)

        step_size = 1. / (num_steps - 1)

        alphas = np.arange(0., 1. + step_size, step_size)

        chars, probs = follow_latent_trajectory(alphas)

        out = OrderedDict()

        out['follow_traj_X_viterbi'] = chars
        out['follow_traj_probs_viterbi'] = probs

        for key, value in out.items():
            np.save(os.path.join(self.out_dir, key + '.npy'), value)
