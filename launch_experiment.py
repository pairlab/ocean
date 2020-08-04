import os
import pathlib
import numpy as np
import click
import json
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, VRNNEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
import random
import pickle
import pdb
import sys

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_dim(s):
    a, b, c, d, e = s.split('.')
    return [int(a), int(b), int(c), int(d), int(e)]

def gpu_optimizer(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

def experiment(variant):
    print (variant['env_name'])
    print (variant['env_params'])
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    cont_latent_dim, num_cat, latent_dim, num_dir, dir_latent_dim = read_dim(variant['global_latent'])
    r_cont_dim, r_n_cat, r_cat_dim, r_n_dir, r_dir_dim = read_dim(variant['vrnn_latent'])
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    glob = variant['algo_params']['glob']
    rnn = variant['rnn']
    vrnn_latent = variant['vrnn_latent']
    encoder_model = MlpEncoder
    if recurrent:
        if variant['vrnn_constraint'] == 'logitnormal':
            output_size = r_cont_dim * 2 + r_n_cat * r_cat_dim + r_n_dir * r_dir_dim * 2
        else:
            output_size = r_cont_dim * 2 + r_n_cat * r_cat_dim + r_n_dir * r_dir_dim
        if variant['rnn_sample'] == 'batch_sampling':
            if variant['algo_params']['use_next_obs']:
                input_size = (2 * obs_dim + action_dim + reward_dim) * variant['temp_res']
            else:
                input_size = (obs_dim + action_dim + reward_dim) * variant['temp_res']
        else:
            if variant['algo_params']['use_next_obs']:
                input_size = (2 * obs_dim + action_dim + reward_dim)
            else:
                input_size = (obs_dim + action_dim + reward_dim)
        if rnn == 'rnn':
            recurrent_model = RecurrentEncoder
            recurrent_context_encoder = recurrent_model(
                hidden_sizes=[net_size, net_size, net_size],
                input_size=input_size,
                output_size = output_size
            )
        elif rnn == 'vrnn':
            recurrent_model = VRNNEncoder
            recurrent_context_encoder = recurrent_model(
                hidden_sizes=[net_size, net_size, net_size],
                input_size=input_size,
                output_size=output_size, 
                temperature=variant['temperature'],
                vrnn_latent=variant['vrnn_latent'],
                vrnn_constraint=variant['vrnn_constraint'],
                r_alpha=variant['vrnn_alpha'],
                r_var=variant['vrnn_var'],
            )

    else:
        recurrent_context_encoder = None

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if glob:
        if dir_latent_dim > 0 and variant['constraint'] == 'logitnormal':
            output_size = cont_latent_dim * 2 + num_cat * latent_dim + num_dir * dir_latent_dim * 2
        else:
            output_size = cont_latent_dim * 2 + num_cat * latent_dim + num_dir * dir_latent_dim
        if variant['algo_params']['use_next_obs']:
            input_size = 2 * obs_dim + action_dim + reward_dim
        else:
            input_size = obs_dim + action_dim + reward_dim
        global_context_encoder = encoder_model(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=input_size,
            output_size=output_size, 
        )
    else:
        global_context_encoder = None      
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim, 
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,  
        output_size=1,
    )
    target_qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,  
        output_size=1,
    )
    target_qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim, 
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim, 
        latent_dim=latent_dim*num_cat + cont_latent_dim + dir_latent_dim*num_dir \
                        + r_n_cat * r_cat_dim + r_cont_dim + r_n_dir * r_dir_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        global_context_encoder,
        recurrent_context_encoder,
        variant['global_latent'],
        variant['vrnn_latent'],
        policy,
        variant['temperature'],
        variant['unitkl'],
        variant['alpha'],
        variant['constraint'],
        variant['vrnn_constraint'],
        variant['var'],
        variant['vrnn_alpha'],
        variant['vrnn_var'],
        rnn,
        variant['temp_res'],
        variant['rnn_sample'],
        variant['weighted_sample'],
        **variant['algo_params']
    )
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        with open(os.path.join(path, 'extra_data.pkl'), 'rb') as f:
            extra_data = pickle.load(f)
            variant['algo_params']['start_epoch'] = extra_data['epoch'] + 1
            replay_buffer = extra_data['replay_buffer']
            enc_replay_buffer = extra_data['enc_replay_buffer']
            variant['algo_params']['_n_train_steps_total'] = extra_data['_n_train_steps_total']
            variant['algo_params']['_n_env_steps_total'] = extra_data['_n_env_steps_total']
            variant['algo_params']['_n_rollouts_total'] = extra_data['_n_rollouts_total']
    else:
        replay_buffer=None
        enc_replay_buffer=None

    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, target_qf1, target_qf2],
        latent_dim=latent_dim,
        replay_buffer=replay_buffer,
        enc_replay_buffer=enc_replay_buffer,
        temp_res=variant['temp_res'],
        rnn_sample=variant['rnn_sample'],
        **variant['algo_params']
    )

    if variant['path_to_weights'] is not None: 
        path = variant['path_to_weights']
        if recurrent_context_encoder != None:
            recurrent_context_encoder.load_state_dict(torch.load(os.path.join(path, 'recurrent_context_encoder.pth')))
        if global_context_encoder != None:
            global_context_encoder.load_state_dict(torch.load(os.path.join(path, 'global_context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        target_qf1.load_state_dict(torch.load(os.path.join(path, 'target_qf1.pth')))
        target_qf2.load_state_dict(torch.load(os.path.join(path, 'target_qf2.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    if ptu.gpu_enabled():
        algorithm.to()

    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))
    exp_id = 'debug' if DEBUG else None
    if variant.get('log_name', "") == "":
        log_name = variant['env_name']
    else:
        log_name = variant['log_name']
    experiment_log_dir = setup_logger(log_name, \
                            variant=variant, \
                            exp_id=exp_id, \
                            base_log_dir=variant['util_params']['base_log_dir'], \
                            config_log_dir=variant['util_params']['config_log_dir'], \
                            log_dir=variant['util_params']['log_dir'])
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    env.save_all_tasks(experiment_log_dir)

    if variant['eval']:
        algorithm._try_to_eval(0, eval_all=True, eval_train_offline=False, animated=True)
    else:
        algorithm.train()

def deep_update_dict(fr, to):
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True, default=False)
@click.option('--seed', default=0)
@click.option('--kl_anneal', default="none", help="none or mono or cycle, annealing of the KL loss")
@click.option('--temperature', default=0.33, help="temperature as in gumbel softmax")
@click.option('--logdir', default='output', help="prefix of the log dir")
@click.option('--kl_lambda', default=1., help="weight of the KL loss")
@click.option('--unitkl', is_flag=True, default=False, help="KL loss on the accumulated global context variables or on each single global context variable")
@click.option('--n_iteration', default=500, help='number of iterations')
@click.option('--alpha', default=0.7)
@click.option('--constraint', default='dirichlet', help='dirichlet or logitnormal for global encoder')
@click.option('--var', default=2.5)
@click.option('--eval', is_flag=True, default=False)
@click.option('--path_to_weights', default=None)
@click.option('--recurrent', is_flag=True, default=False, help='Use local encoder or not')
@click.option('--vrnn_latent', default='2.0.0.2.4', help='gaus-dim.num-cat.cat-dim.num-dir.dir-dim')
@click.option('--global_latent', default='2.0.0.2.4', help='gaus-dim.num-cat.cat-dim.num-dir.dir-dim')
@click.option('--rnn', default='rnn', help='rnn or vrnn or None, architecture of the local encoder')
@click.option('--traj_batch_size', default=16, help='Number of trajectories sampled in one update')
@click.option('--vrnn_constraint', default='dirichlet', help="logitnormal or dirichlet")
@click.option('--vrnn_alpha', default=0.7)
@click.option('--vrnn_var', default=2.5)
@click.option('--temp_res', default=10, help="Temporal resolution")
@click.option('--rnn_sample', default="full", help="full or full_wo_sampling or single_sampling or batch_sampling") # full: infer posterior and update local variable each step; full_wo_sampling: infer posterior each step but update local variable every temp_res steps; single_sampling: infer posterior every temp_res steps using only the contexts in the previous step; batch_sampling: infer posterior every temp_res steps using previous temp_res contexts. 
@click.option('--resample_in_traj', is_flag=True, default=False, help='resample global context variable in one trajectory, can be used in model with only global encoder')
@click.option('--weighted_sample', is_flag=True, default=False, help="When calculating the global posterior, do we use weighted product of the PDF or not")
@click.option('--use_next_obs', is_flag=True, default=False, help='Context uses next observation or not')
def main(config, gpu, debug, seed, kl_anneal, temperature, logdir, kl_lambda, 
            unitkl, 
            n_iteration, alpha, 
            constraint, 
            var, 
            eval, path_to_weights, 
            recurrent, vrnn_latent, global_latent, rnn, traj_batch_size, vrnn_constraint, 
            vrnn_alpha, vrnn_var, temp_res, rnn_sample, resample_in_traj, 
            weighted_sample, use_next_obs, 
            ):
    cont_latent_size, num_cat, latent_size, num_dir, dir_latent_size = read_dim(global_latent)
    glob = latent_size * num_cat + cont_latent_size + dir_latent_size * num_dir > 0
    if resample_in_traj:
        assert glob
    assert kl_anneal in ['none', 'mono', 'cycle']
    if not recurrent:
        vrnn_latent = '0.0.0.0.0'
        rnn = 'None'
        traj_batch_size = -1
        vrnn_constraint = None
        vrnn_alpha = None
        vrnn_var = None
        if not resample_in_traj:
            temp_res = None
        rnn_sample = None
    r_cont_dim, r_n_cat, r_cat_dim, r_n_dir, r_dir_dim = read_dim(vrnn_latent)
    if recurrent:
        temp_res = int(temp_res)
        assert rnn_sample in ["full", "full_wo_sampling", "single_sampling", "batch_sampling"]
        if rnn_sample == 'full':
            temp_res = 1
        if r_dir_dim > 0:
            assert vrnn_constraint in ['logitnormal', 'dirichlet']
            if vrnn_constraint == 'logitnormal':
                vrnn_alpha = None
            else:
                vrnn_var = None
        else:
            vrnn_alpha = None
            vrnn_var = None
            vrnn_constraint = None

    if resample_in_traj:
        temp_res = int(temp_res)
    if latent_size == 0:
        num_cat = 0
    if dir_latent_size == 0:
        num_dir = 0
    if dir_latent_size > 0:
        assert constraint in ['dirichlet', 'logitnormal']
        if constraint == 'logitnormal':
            alpha = None
        else:
            var = None
    else:
        constraint = None
        alpha = None
        var = None

    set_global_seeds(seed)
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    if gpu != -1:
        variant['util_params']['gpu_id'] = gpu
    else:
        variant['util_params']['use_gpu'] = False
    variant['seed'] = seed
    variant['temperature'] = temperature
    variant['env_params']['seed'] = seed

    variant['algo_params']['kl_anneal'] = kl_anneal
    variant['util_params']['base_log_dir'] = logdir
    variant["algo_params"]['kl_lambda'] = kl_lambda
    variant['algo_params']['use_next_obs'] = use_next_obs
    variant['unitkl'] = unitkl
    variant['alpha'] = alpha
    variant['var'] = var
    variant['constraint'] = constraint
    variant['eval'] = eval
    variant['path_to_weights'] = path_to_weights
    variant['algo_params']['recurrent'] = recurrent 
    variant['algo_params']['glob'] = glob 

    variant['vrnn_latent'] = vrnn_latent 
    variant['global_latent'] = global_latent
    variant['rnn'] = rnn
    variant['algo_params']['traj_batch_size'] = traj_batch_size
    variant['vrnn_constraint'] = vrnn_constraint
    variant['vrnn_alpha'] = vrnn_alpha
    variant['vrnn_var'] = vrnn_var
    variant['weighted_sample'] = weighted_sample
    variant['util_params']['log_dir'] = None

    variant['temp_res'] = temp_res
    variant['rnn_sample'] = rnn_sample
    variant['algo_params']['resample_in_traj'] = resample_in_traj

    save_dir = 'global-{}-local-{}-{}-{}-temp-{}-{}'.format(global_latent, vrnn_latent, vrnn_constraint, rnn, temp_res, rnn_sample)
    if resample_in_traj:
        save_dir += '-resample'
    if weighted_sample:
        save_dir += '-ws'
    save_dir = os.path.join(save_dir, 'seed-%s'%seed)
    variant['util_params']['config_log_dir'] = save_dir
    variant['meta'] = None

    if eval:
        variant['util_params']['config_log_dir'] = os.path.join('eval', variant['util_params']['config_log_dir'])  
    variant['util_params']['debug'] = debug
    variant['algo_params']['num_iterations'] = int(n_iteration)
    variant['algo_params']['debug'] = debug
    experiment(variant)

if __name__ == "__main__":
    main()

