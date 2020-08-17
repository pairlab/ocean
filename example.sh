#!/bin/bash

## Humanoid
# OCEAN w/ rnn
python3.5 launch_experiment.py ./configs/humanoid-multi-dir.json --logdir log --gpu 0 --kl_lambda 10 --n_iteration 500 --seed 4 --global_latent 0.0.0.2.3 --constraint dirichlet --alpha 0.8 --unitkl --recurrent --vrnn_latent 0.0.0.2.3 --rnn rnn --temp_res 20 --rnn_sample batch_sampling --traj_batch_size 1 --vrnn_constraint dirichlet --vrnn_alpha 0.8 --weighted_sample

# OCEAN w/ vrnn (can be slow, run rnn version for a quick try)
python3.5 launch_experiment.py ./configs/humanoid-multi-dir.json --logdir log --gpu 0 --kl_lambda 10 --n_iteration 500 --seed 4 --global_latent 0.0.0.2.3 --constraint dirichlet --alpha 0.8 --unitkl --recurrent --vrnn_latent 0.0.0.2.3 --rnn vrnn --temp_res 50 --rnn_sample single_sampling --traj_batch_size 1 --vrnn_constraint dirichlet --vrnn_alpha 0.8 --weighted_sample

# PEARL
python3.5 launch_experiment.py ./configs/humanoid-multi-dir.json --logdir log --gpu 0 --kl_lambda 10 --n_iteration 500 --seed 4 --global_latent 12.0.0.0.0 --weighted_sample


## Cheetah
# OCEAN w/ rnn
python3.5 launch_experiment.py ./configs/cheetah-multi-dir-complex.json --logdir log --gpu 0 --kl_lambda 10 --n_iteration 500 --seed 4 --global_latent 0.0.0.2.3 --constraint dirichlet --alpha 0.8 --recurrent --vrnn_latent 0.0.0.2.3 --rnn rnn --temp_res 100 --rnn_sample batch_sampling --traj_batch_size 1 --vrnn_constraint dirichlet --vrnn_alpha 0.8 --weighted_sample

# OCEAN w/ vrnn
python3.5 launch_experiment.py ./configs/cheetah-multi-dir-complex.json --logdir log --gpu 0 --kl_lambda 10 --n_iteration 500 --seed 4 --global_latent 0.0.0.2.3 --constraint dirichlet --alpha 0.8 --recurrent --vrnn_latent 0.0.0.2.3 --rnn vrnn --temp_res 100 --rnn_sample batch_sampling --traj_batch_size 1 --vrnn_constraint dirichlet --vrnn_alpha 0.8 --weighted_sample

# PEARL
python3.5 launch_experiment.py ./configs/cheetah-multi-dir-complex.json --logdir log --gpu 0 --kl_lambda 10 --n_iteration 500 --seed 4 --global_latent 12.0.0.0.0 --weighted_sample
