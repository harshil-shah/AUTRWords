from model.distributions import GaussianDiagonal, Categorical
from model.generative_models import GenAUTRWords as GenerativeModel
from model.recognition_models import RecMLP as RecognitionModel

import json
import os
import sys

import lasagne

from trainers.sgvb import SGVBWords as SGVB
from run import RunWords as Run

import numpy as np
np.set_printoptions(threshold=1000000)


sys.setrecursionlimit(5000000)


main_dir = sys.argv[1]
out_dir = sys.argv[2]


dataset = 'WikipediaWestbury/processed_vocab20000_2to200_limunk0.0'

with open(os.path.join(main_dir, '../_datasets', dataset, 'valid_vocab.txt'), 'r') as f:
    valid_vocab = json.loads(f.read())


solver = SGVB


vocab_size = len(valid_vocab)
restrict_min_length = 1
restrict_max_length = 10
train_prop = 0.9

d_z = 50
d_emb = 300

gen_nn_kwargs = {
    'rnn_depth': 2,
    'rnn_hid_units': 500,
    'rnn_hid_nonlinearity': lasagne.nonlinearities.elu,
    'rnn_time_steps': 5,
}

rec_nn_kwargs = {
    'depth': 4,
    'hid_units': 1000,
    'hid_nonlinearity': lasagne.nonlinearities.tanh,
    # 'rnn_hid_dim': 1000,
    # 'final_depth': 3,
    # 'final_hid_units': 1000,
    # 'final_hid_nonlinearity': lasagne.nonlinearities.elu,
}

solver_kwargs = {'generative_model': GenerativeModel,
                 'recognition_model': RecognitionModel,
                 'vocab_size': vocab_size,
                 'gen_nn_kwargs': gen_nn_kwargs,
                 'rec_nn_kwargs': rec_nn_kwargs,
                 'z_dim': d_z,
                 'embedding_dim': d_emb,
                 'dist_z_gen': GaussianDiagonal,
                 'dist_x_gen': Categorical,
                 'dist_z_rec': GaussianDiagonal,
                 }

pre_trained = False
load_param_dir = 'code_outputs/2017_06_26_20_07_56'

train = True

training_iterations = 250000
training_batch_size = 200
training_num_samples = 1
warm_up = 25000

grad_norm_constraint = None
update = lasagne.updates.adam
update_kwargs = {'learning_rate': 0.0001}

val_freq = 1000
val_batch_size = 100
val_num_samples = 1

save_params_every = 50000


generate_output_prior = False
generate_output_posterior = False

impute_missing_chars = False
drop_rate = 0.2
missing_chars_iterations = 500

find_best_matches = False
num_best_matches = 20
best_matches_batch_size = 100

follow_latent_trajectory = False
latent_trajectory_steps = 10

num_outputs = 100


test = False
test_batch_size = 500
test_num_samples = 100
test_sub_sample_size = 10


if __name__ == '__main__':

    run = Run(dataset=dataset, valid_vocab=valid_vocab, solver=solver,
              solver_kwargs=solver_kwargs, main_dir=main_dir, out_dir=out_dir, pre_trained=pre_trained,
              load_param_dir=load_param_dir, restrict_min_length=restrict_min_length,
              restrict_max_length=restrict_max_length, train_prop=train_prop)

    if train:
        run.train(n_iter=training_iterations, batch_size=training_batch_size, num_samples=training_num_samples,
                  grad_norm_constraint=grad_norm_constraint, update=update, update_kwargs=update_kwargs,
                  val_freq=val_freq, val_batch_size=val_batch_size, val_num_samples=val_num_samples, warm_up=warm_up,
                  save_params_every=save_params_every)

    if generate_output_prior or generate_output_posterior:
        run.generate_output(prior=generate_output_prior, posterior=generate_output_posterior, num_outputs=num_outputs)

    if impute_missing_chars:
        run.impute_missing_chars(num_outputs=num_outputs, drop_rate=drop_rate, num_iterations=missing_chars_iterations)

    if find_best_matches:
        run.find_best_matches(num_outputs=num_outputs, num_matches=num_best_matches, batch_size=best_matches_batch_size)

    if follow_latent_trajectory:
        run.follow_latent_trajectory(num_samples=num_outputs, num_steps=latent_trajectory_steps)

    if test:
        run.test(test_batch_size, test_num_samples, test_sub_sample_size)
