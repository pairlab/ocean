import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu

import pdb
import os

eps = 1e-11

def _product_of_categorical_all(z_means):
    z_means = torch.log(z_means+eps)
    z_mean = torch.sum(z_means, dim=-2)
    cc = torch.max(z_mean).detach()
    z_mean -= cc
    z_mean = torch.exp(z_mean)
    return F.normalize(z_mean, p=1, dim=-1)

def _weighted_product_of_categorical_all(z_means):
    z_means = torch.log(z_means+eps)
    z_mean = torch.mean(z_means, dim=-2)
    cc = torch.max(z_mean).detach()
    z_mean -= cc
    z_mean = torch.exp(z_mean)
    return F.normalize(z_mean, p=1, dim=-1)

def _product_of_gaussians_all(mus, sigmas_squared):
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=-2)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=-2)
    return mu, sigma_squared

def _weighted_product_of_gaussians_all(mus, sigmas_squared):
    n = mus.shape[-2]
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = n / torch.sum(torch.reciprocal(sigmas_squared), dim=-2)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=-2) / n
    return mu, sigma_squared

def _weighted_product_of_dirichlet_all(alphas):
    return torch.mean(alphas, dim=-2)

def read_dim(s):
    a, b, c, d, e = s.split('.')
    return [int(a), int(b), int(c), int(d), int(e)]

class PEARLAgent(nn.Module):

    def __init__(self,
                 global_context_encoder,
                 recurrent_context_encoder,
                 global_latent,
                 vrnn_latent,
                 policy,
                 temperature,
                 unitkl,
                 alpha,
                 g_constraint,
                 r_constraint,
                 var,
                 r_alpha,
                 r_var,
                 rnn,
                 temp_res,
                 rnn_sample,
                 weighted_sample,
                 **kwargs
    ):
        super().__init__()
        self.g_cont_dim, self.g_n_cat, self.g_cat_dim, self.g_n_dir, self.g_dir_dim = read_dim(global_latent)
        if recurrent_context_encoder != None:
            self.r_cont_dim, self.r_n_cat, self.r_cat_dim, self.r_n_dir, self.r_dir_dim = read_dim(vrnn_latent)

        self.global_context_encoder = global_context_encoder
        self.recurrent_context_encoder = recurrent_context_encoder
        self.policy = policy
        self.temperature = temperature
        self.unitkl = unitkl

        self.g_constraint = g_constraint # global dirichlet type
        self.r_constraint = r_constraint # local dirichlet type
        self.g_alpha = alpha
        self.g_var = var
        self.r_alpha = r_alpha
        self.r_var = r_var
        self.rnn = rnn

        self.weighted_sample = weighted_sample

        self.temp_res = temp_res
        self.rnn_sample = rnn_sample
        self.n_global, self.n_local, self.n_infer = 0, 0, 0

        self.recurrent = kwargs['recurrent']
        self.glob = kwargs['glob']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs = kwargs['use_next_obs']
        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        if self.glob:
            self.register_buffer('z', torch.zeros(1, self.g_cont_dim + self.g_cat_dim * self.g_n_cat + self.g_dir_dim * self.g_n_dir))
            if self.g_cat_dim > 0:
                self.register_buffer('z_means', torch.zeros(1, self.g_cat_dim))
            if self.g_cont_dim > 0:
                self.register_buffer('z_c_means', torch.zeros(1, self.g_cont_dim))
                self.register_buffer('z_c_vars', torch.ones(1, self.g_cont_dim))
            if self.g_dir_dim > 0:
                if self.g_constraint == 'logitnormal':
                    self.register_buffer('z_d_means', torch.zeros(1, self.g_dir_dim))
                    self.register_buffer('z_d_vars', torch.ones(1, self.g_dir_dim))
                elif self.g_constraint == 'dirichlet':
                    self.register_buffer('z_d_means', torch.zeros(1, self.g_dir_dim))
        
        if self.recurrent:
            self.register_buffer('seq_z', torch.zeros(1, self.r_cont_dim + self.r_cat_dim * self.r_n_cat + self.r_dir_dim * self.r_n_dir))
            z_cat_prior, z_cont_prior, z_dir_prior = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
            if self.r_cat_dim > 0:
                self.register_buffer('seq_z_cat', torch.zeros(1, self.r_cat_dim))
                self.seq_z_next_cat = None
                z_cat_prior = ptu.ones(self.r_cat_dim * self.r_n_cat) / self.r_cat_dim
            if self.r_dir_dim > 0:
                if self.r_constraint == 'logitnormal':
                    self.register_buffer('seq_z_dir_mean', torch.zeros(1, self.r_dir_dim))
                    self.register_buffer('seq_z_dir_var', torch.ones(1, self.r_dir_dim))
                    self.seq_z_next_dir_mean = None
                    self.seq_z_next_dir_var = None
                    z_dir_prior_mean = ptu.zeros(self.r_n_dir * self.r_dir_dim)
                    z_dir_prior_var = ptu.ones(self.r_n_dir * self.r_dir_dim) * self.r_var
                    z_dir_prior = torch.cat([z_dir_prior_mean, z_dir_prior_var])
                elif self.r_constraint == 'dirichlet':
                    self.register_buffer('seq_z_dir', torch.zeros(1, self.r_dir_dim))
                    self.seq_z_next_dir = None
                    z_dir_prior = ptu.ones(self.r_n_dir * self.r_dir_dim) * self.r_alpha
            if self.r_cont_dim > 0:
                self.register_buffer('seq_z_cont_mean', torch.zeros(1, self.r_cont_dim))
                self.register_buffer('seq_z_cont_var', torch.zeros(1, self.r_cont_dim))
                self.seq_z_next_cont_mean = None
                self.seq_z_next_cont_var = None
                z_cont_prior = torch.cat([ptu.zeros(self.r_cont_dim), ptu.ones(self.r_cont_dim)])
            self.seq_z_prior = torch.cat([z_cat_prior, z_cont_prior, z_dir_prior])

        self.clear_z()
        

    def clear_z(self, num_tasks=1, batch_size=1, traj_batch_size=1): 
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        if self.glob:
            if self.g_cat_dim > 0:
                self.z_means = ptu.ones(num_tasks * self.g_n_cat, self.g_cat_dim)/self.g_cat_dim
            if self.g_cont_dim > 0:
                self.z_c_means = ptu.zeros(num_tasks, self.g_cont_dim)
                self.z_c_vars = ptu.ones(num_tasks, self.g_cont_dim)
            if self.g_dir_dim > 0:
                if self.g_constraint == 'logitnormal':
                    self.z_d_means = ptu.zeros(num_tasks * self.g_n_dir, self.g_dir_dim)
                    self.z_d_vars = ptu.ones(num_tasks * self.g_n_dir, self.g_dir_dim)*self.g_var
                else:
                    self.z_d_means = ptu.ones(num_tasks * self.g_n_dir, self.g_dir_dim)*self.g_alpha                

            self.sample_z()

        if self.recurrent:
            if self.r_cat_dim > 0:
                self.seq_z_cat = ptu.ones(num_tasks * batch_size * self.r_n_cat, self.r_cat_dim) / self.r_cat_dim
                self.seq_z_next_cat = None
            if self.r_cont_dim > 0:
                self.seq_z_cont_mean = ptu.zeros(num_tasks * batch_size, self.r_cont_dim)
                self.seq_z_cont_var = ptu.ones(num_tasks * batch_size, self.r_cont_dim)
                self.seq_z_next_cont_mean = None
                self.seq_z_next_cont_var = None
            if self.r_dir_dim > 0:
                if self.r_constraint == 'logitnormal':
                    self.seq_z_dir_mean = ptu.zeros(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim)
                    self.seq_z_dir_var = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_var
                    self.seq_z_next_dir_mean = None
                    self.seq_z_next_dir_var = None
                elif self.r_constraint == 'dirichlet':
                    self.seq_z_dir = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_alpha
                    self.seq_z_next_dir = None

            self.sample_sequence_z()


        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        if self.global_context_encoder != None:
            self.global_context_encoder.reset(num_tasks)
        if self.recurrent_context_encoder != None:
            self.recurrent_context_encoder.reset(num_tasks*traj_batch_size)

    def detach_z(self):
        ''' disable backprop through z '''
        if self.glob:
            self.z = self.z.detach()
        if self.recurrent:
            self.recurrent_context_encoder.hn = self.recurrent_context_encoder.hn.detach()
            self.recurrent_context_encoder.cn = self.recurrent_context_encoder.cn.detach()
            self.seq_z = self.seq_z.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self): 
        ''' compute KL( q(z|c) || r(z) ) '''
        kl_div_cont, kl_div_disc, kl_div_dir = ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean()
        kl_div_seq_cont, kl_div_seq_disc, kl_div_seq_dir = ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean()

        if self.glob:
            if self.g_cat_dim > 0:
                if self.unitkl:
                    kl_div_disc = torch.sum(self.z_means_all*torch.log((self.z_means_all+eps)*self.g_cat_dim))
                else:
                    kl_div_disc = torch.sum(self.z_means*torch.log((self.z_means+eps)*self.g_cat_dim))
            if self.g_dir_dim > 0:
                if self.g_constraint == 'dirichlet':
                    prior = torch.distributions.Dirichlet(ptu.ones(self.g_dir_dim)*self.g_alpha)
                    if self.unitkl:
                        posteriors = torch.distributions.Dirichlet(self.z_d_means_all)
                    else:
                        posteriors = torch.distributions.Dirichlet(self.z_d_means)
                    kl_div_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior))
                elif self.g_constraint == 'logitnormal':
                    prior = torch.distributions.Normal(ptu.zeros(self.g_dir_dim), ptu.ones(self.g_dir_dim)*np.sqrt(self.g_var))
                    if self.unitkl:
                        posteriors = torch.distributions.Normal(self.z_d_means_all, torch.sqrt(self.z_d_vars_all))
                    else:
                        posteriors = torch.distributions.Normal(self.z_d_means, torch.sqrt(self.z_d_vars))
                    kl_div_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior))
            if self.g_cont_dim > 0:
                if self.unitkl:
                    kl_div_cont = torch.sum(0.5*(-torch.log(self.z_c_vars_all)+self.z_c_vars_all+self.z_c_means_all*self.z_c_means_all-1)) 
                else:
                    kl_div_cont = torch.sum(0.5*(-torch.log(self.z_c_vars)+self.z_c_vars+self.z_c_means*self.z_c_means-1)) 

        if self.recurrent:
            if self.rnn == 'rnn':
                if self.r_cat_dim > 0:
                    assert type(self.seq_z_next_cat) != type(None)
                    kl_div_seq_disc = torch.sum(self.seq_z_cat * torch.log((self.seq_z_cat + eps) * self.r_cat_dim)) \
                                        + torch.sum(self.seq_z_next_cat * torch.log((self.seq_z_next_cat + eps) * self.r_cat_dim))
                if self.r_dir_dim > 0:
                    if self.r_constraint == 'dirichlet':
                        assert type(self.seq_z_next_dir) != type(None)
                        prior = torch.distributions.Dirichlet(ptu.ones(self.r_dir_dim) * self.r_alpha)
                        posteriors = torch.distributions.Dirichlet(self.seq_z_dir)
                        posteriors_next = torch.distributions.Dirichlet(self.seq_z_next_dir)
                        kl_div_seq_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior)) \
                                            + torch.sum(torch.distributions.kl.kl_divergence(posteriors_next, prior))
                    elif self.r_constraint == 'logitnormal':
                        assert type(self.seq_z_next_dir_mean) != type(None)
                        prior = torch.distributions.Normal(ptu.zeros(self.r_dir_dim), ptu.ones(self.r_dir_dim)*np.sqrt(self.r_var))
                        posteriors = torch.distributions.Normal(self.seq_z_dir_mean, torch.sqrt(self.seq_z_dir_var))
                        posteriors_next = torch.distributions.Normal(self.seq_z_next_dir_mean, torch.sqrt(self.seq_z_next_dir_var))
                        kl_div_seq_dir = torch.sum(torch.distributions.kl.kl_divergence(posteriors, prior)) \
                                            + torch.sum(torch.distributions.kl.kl_divergence(posteriors_next, prior))
                if self.r_cont_dim > 0:
                    kl_div_seq_cont = torch.sum(0.5*(-torch.log(self.seq_z_cont_var)+self.seq_z_cont_var+self.seq_z_cont_mean*self.seq_z_cont_mean-1)) \
                                        + torch.sum(0.5*(-torch.log(self.seq_z_next_cont_var)+self.seq_z_next_cont_var+self.seq_z_next_cont_mean*self.seq_z_next_cont_mean-1)) 
            elif self.rnn == 'vrnn':
                kl_div_seq_disc, kl_div_seq_cont, kl_div_seq_dir = self.recurrent_context_encoder.compute_kl_div()


        return kl_div_disc, kl_div_cont, kl_div_dir, kl_div_seq_disc, kl_div_seq_cont, kl_div_seq_dir

    def infer_posterior(self, context, ff=False): 
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.global_context_encoder(context)
        if self.g_dir_dim > 0 and self.g_constraint == 'dirichlet':
            params = params.view(context.size(0), -1, self.g_n_cat*self.g_cat_dim + 2*self.g_cont_dim + self.g_n_dir*self.g_dir_dim)
        else:                
            params = params.view(context.size(0), -1, self.g_n_cat*self.g_cat_dim + 2*self.g_cont_dim + self.g_n_dir*self.g_dir_dim*2)

        if self.g_cat_dim > 0:
            params_disc = params[..., :self.g_n_cat*self.g_cat_dim]
            params_disc = params_disc.view(context.size(0), -1, self.g_n_cat, self.g_cat_dim)
            params_disc = params_disc.transpose(1, 2)
            mu = F.softmax(params_disc, dim=-1)
            if self.unitkl:
                self.z_means_all = torch.reshape(mu, [-1, self.g_cat_dim])
            if self.weighted_sample:
                self.z_means = _weighted_product_of_categorical_all(mu).view(-1, self.g_cat_dim)
            else:
                self.z_means = _product_of_categorical_all(mu).view(-1, self.g_cat_dim)
        
        if self.g_cont_dim > 0:
            params_cont = params[..., self.g_n_cat*self.g_cat_dim:self.g_n_cat*self.g_cat_dim+2*self.g_cont_dim]
            mu_c = params_cont[..., :self.g_cont_dim]
            sigma_squared_c = F.softplus(params_cont[..., self.g_cont_dim:])
            if self.unitkl:
                self.z_c_means_all = torch.reshape(mu_c, [-1, self.g_cont_dim])
                self.z_c_vars_all = torch.reshape(sigma_squared_c, [-1, self.g_cont_dim])
            if self.weighted_sample:
                self.z_c_means, self.z_c_vars = _weighted_product_of_gaussians_all(mu_c, sigma_squared_c)
            else:
                self.z_c_means, self.z_c_vars = _product_of_gaussians_all(mu_c, sigma_squared_c)

        if self.g_dir_dim > 0 and self.g_constraint == 'logitnormal':
            params_dir = params[..., self.g_n_cat*self.g_cat_dim+2*self.g_cont_dim:]
            params_dir = params_dir.view(context.size(0), -1, self.g_n_dir, self.g_dir_dim*2)
            params_dir = params_dir.transpose(1, 2)
            mu_d = params_dir[..., :self.g_dir_dim]
            sigma_squared_d = F.softplus(params_dir[..., self.g_dir_dim:])
            if self.unitkl:
                self.z_d_means_all = torch.reshape(mu_d, [-1, self.g_dir_dim])
                self.z_d_vars_all = torch.reshape(sigma_squared_d, [-1, self.g_dir_dim])
            if self.weighted_sample:
                self.z_d_means, self.z_d_vars = _weighted_product_of_gaussians_all(mu_d, sigma_squared_d)
            else:
                self.z_d_means, self.z_d_vars = _product_of_gaussians_all(mu_d, sigma_squared_d)
            self.z_d_means = self.z_d_means.view(-1, self.g_dir_dim)
            self.z_d_vars = self.z_d_vars.view(-1, self.g_dir_dim)

        if self.g_dir_dim > 0 and self.g_constraint == 'dirichlet':
            params_dir = params[..., self.g_n_cat*self.g_cat_dim+2*self.g_cont_dim:]
            params_dir = params_dir.view(context.size(0), -1, self.g_n_dir, self.g_dir_dim)
            params_dir = F.softplus(params_dir.transpose(1, 2))
            if self.unitkl:
                # self.z_d_means_all = params_dir.view(-1, self.g_dir_dim)
                self.z_d_means_all = torch.reshape(params_dir, [-1, self.g_dir_dim])
            if self.weighted_sample:
                self.z_d_means = _weighted_product_of_dirichlet_all(params_dir)
            else:
                assert False, "Global dirichlet parameterization must be weighted sample"
            self.z_d_means = self.z_d_means.view(-1, self.g_dir_dim)

        self.sample_z()

    def sample_z(self):
        z, z_c, z_d = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
        if self.g_cat_dim > 0:
            gumbel = torch.distributions.Gumbel(ptu.FloatTensor([0]), ptu.FloatTensor([1.0])).sample(self.z_means.size()).squeeze(-1)
            log_z = torch.log(self.z_means+eps)
            logit = (log_z + gumbel) / self.temperature
            z = F.softmax(logit, dim=1).view(-1, self.g_n_cat, self.g_cat_dim).view(-1, self.g_n_cat * self.g_cat_dim)
        if self.g_cont_dim > 0:
            normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(self.z_c_means.size()).squeeze(-1)
            z_c = self.z_c_means + torch.sqrt(self.z_c_vars)*normal
        if self.g_dir_dim > 0:
            if self.g_constraint == 'dirichlet':
                z_d = torch.distributions.Dirichlet(self.z_d_means).rsample()\
                    .view(-1, self.g_n_dir, self.g_dir_dim).view(-1, self.g_n_dir * self.g_dir_dim)
            elif self.g_constraint == 'logitnormal':
                normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(self.z_d_means.size()).squeeze(-1)
                z_d = F.softmax(self.z_d_means + torch.sqrt(self.z_d_vars)*normal, dim=-1)\
                        .view(-1, self.g_n_dir, self.g_dir_dim).view(-1, self.g_n_dir * self.g_dir_dim)

        self.z = torch.cat([z, z_c, z_d], dim=-1)

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z, seq_z = ptu.FloatTensor(), ptu.FloatTensor()
        if self.glob:
            z = self.z
        if self.recurrent:
            seq_z = self.seq_z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z, seq_z], dim=1)

        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, trajectories, indices_in_trajs, do_inference, compute_for_next, is_next):
        ''' given context, get statistics under the current policy of a set of observations '''
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)

        task_z, seq_z = ptu.FloatTensor(), ptu.FloatTensor()
        # self.n_infer += 1
        if do_inference:
            if self.recurrent:
                assert type(self.recurrent_context_encoder) != type(None) and type(trajectories) != type(None) and type(indices_in_trajs) != type(None)
                self.infer_sequence_posterior(trajectories, indices_in_trajs, compute_for_next = compute_for_next)

            if self.glob:
                self.infer_posterior(context)

        if self.recurrent:
            if is_next:
                seq_z = self.seq_z_next
            else:
                seq_z = self.seq_z

        if self.glob:
            task_z = self.z
            task_z = [z.repeat(b, 1) for z in task_z]
            task_z = torch.cat(task_z, dim=0)

        in_ = torch.cat([obs, task_z.detach(), seq_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True) 

        return policy_outputs, task_z, seq_z

    def log_diagnostics(self, eval_statistics): 
        pass

    @property
    def networks(self):
        network_list = []
        if self.glob:
            network_list.append(self.global_context_encoder)
        network_list.append(self.policy)
        if self.recurrent:
            network_list.append(self.recurrent_context_encoder)
        return network_list

    def infer_sequence_posterior(self, trajectories, indices_in_trajs, compute_for_next): 
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        num_tasks, traj_batch, eps_len, input_dim = trajectories.size()
        self.clear_sequence_z(num_tasks=num_tasks, batch_size=traj_batch * indices_in_trajs.size(2), traj_batch_size=traj_batch)
        if self.rnn_sample == 'full':
            params = self.recurrent_context_encoder(trajectories.view(-1, eps_len, input_dim))
        elif self.rnn_sample == 'full_wo_sampling':
            params = self.recurrent_context_encoder(trajectories.view(-1, eps_len, input_dim))
        elif self.rnn_sample == 'single_sampling':
            traj_ranges = [i for i in range(eps_len) if i % self.temp_res == (self.temp_res - 1)]
            tmp_trajectories = trajectories[:, :, traj_ranges, :]
            params = self.recurrent_context_encoder(tmp_trajectories.view(-1, len(traj_ranges), input_dim))
            eps_len = len(traj_ranges)
        elif self.rnn_sample == 'batch_sampling':
            max_len = int(eps_len//self.temp_res*self.temp_res)
            tmp_trajectories = trajectories[:, :, :max_len, :]
            tmp_trajectories = tmp_trajectories.view(num_tasks * traj_batch, max_len // self.temp_res, self.temp_res * input_dim)
            params = self.recurrent_context_encoder(tmp_trajectories)
            eps_len = max_len // self.temp_res

        if self.rnn_sample == 'full':
            if compute_for_next:
                indices_in_trajs_next = indices_in_trajs + 1
        else:
            if compute_for_next:
                indices_in_trajs_next = (indices_in_trajs + 1) // self.temp_res
            indices_in_trajs = indices_in_trajs // self.temp_res
            
        if self.r_constraint == 'logitnormal':
            params = params.view(num_tasks, traj_batch, eps_len, self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim * 2)
        else:
            params = params.view(num_tasks, traj_batch, eps_len, self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim)

        if self.rnn_sample == 'full_wo_sampling':
            traj_ranges = [i for i in range(eps_len) if i % self.temp_res == (self.temp_res - 1)]
            params = params[:, :, traj_ranges, :]

        batch_per_traj = indices_in_trajs.size(2)
        params = torch.cat([self.seq_z_prior.expand(num_tasks, traj_batch, 1, params.size(3)), params], dim=2)

        if self.r_cat_dim > 0:
            params_disc = params[..., :self.r_n_cat * self.r_cat_dim]
            seq_z_cat = torch.gather(params_disc, 2, indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_cat * self.r_cat_dim))
            self.seq_z_cat = F.softmax(seq_z_cat.view(num_tasks * traj_batch * batch_per_traj * self.r_n_cat, self.r_cat_dim), dim=-1)
            if compute_for_next:
                seq_z_next_cat = torch.gather(params_disc, 2, indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_cat * self.r_cat_dim)) 
                self.seq_z_next_cat = F.softmax(seq_z_next_cat.view(num_tasks * traj_batch * batch_per_traj * self.r_n_cat, self.r_cat_dim), dim=-1)
            else:
                self.seq_z_next_cat = None

        if self.r_cont_dim > 0:
            params_cont = params[..., self.r_n_cat * self.r_cat_dim : self.r_n_cat * self.r_cat_dim + 2 * self.r_cont_dim]
            mu_c = params_cont[..., :self.r_cont_dim]
            sigma_squared_c = F.softplus(params_cont[..., self.r_cont_dim:])
            seq_z_cont_mean = torch.gather(mu_c, 2, indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_cont_dim))
            seq_z_cont_var = torch.gather(sigma_squared_c, 2, indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_cont_dim))
            self.seq_z_cont_mean = seq_z_cont_mean.view(num_tasks * traj_batch * batch_per_traj, self.r_cont_dim)
            self.seq_z_cont_var = seq_z_cont_var.view(num_tasks * traj_batch * batch_per_traj, self.r_cont_dim)
            if compute_for_next:
                seq_z_next_cont_mean = torch.gather(mu_c, 2, indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_cont_dim))
                seq_z_next_cont_var = torch.gather(sigma_squared_c, 2, indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_cont_dim))
                self.seq_z_next_cont_mean = seq_z_next_cont_mean.view(num_tasks * traj_batch * batch_per_traj, self.r_cont_dim)
                self.seq_z_next_cont_var = seq_z_next_cont_var.view(num_tasks * traj_batch * batch_per_traj, self.r_cont_dim)
            else:
                self.seq_z_next_cont_var = None
                self.seq_z_next_cont_mean = None

        if self.r_dir_dim > 0 and self.r_constraint == 'logitnormal':
            params_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            mu_d = params_dir[..., :self.r_n_dir * self.r_dir_dim]
            sigma_squared_d = F.softplus(params_dir[..., self.r_dir_dim * self.r_n_dir:])
            seq_z_dir_mean = torch.gather(mu_d, 2, indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_dir * self.r_dir_dim))
            seq_z_dir_var = torch.gather(sigma_squared_d, 2, indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_dir * self.r_dir_dim))
            self.seq_z_dir_mean = seq_z_dir_mean.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim)
            self.seq_z_dir_var = seq_z_dir_var.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim)
            if compute_for_next:
                seq_z_next_dir_mean = torch.gather(mu_d, 2, indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_dir * self.r_dir_dim))
                seq_z_next_dir_var = torch.gather(sigma_squared_d, 2, indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_dir * self.r_dir_dim))
                self.seq_z_next_dir_mean = seq_z_next_dir_mean.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim)
                self.seq_z_next_dir_var = seq_z_next_dir_var.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim)
            else:
                self.seq_z_next_dir_mean = None
                self.seq_z_next_dir_var = None

        if self.r_dir_dim > 0 and self.r_constraint == 'dirichlet':
            params_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            seq_z_dir = torch.gather(params_dir, 2, indices_in_trajs.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_dir * self.r_dir_dim))
            self.seq_z_dir = F.softplus(seq_z_dir.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim))
            if compute_for_next:
                seq_z_next_dir = torch.gather(params_dir, 2, indices_in_trajs_next.unsqueeze(-1).expand(num_tasks, traj_batch, batch_per_traj, self.r_n_dir * self.r_dir_dim))
                self.seq_z_next_dir = F.softplus(seq_z_next_dir.view(num_tasks * traj_batch * batch_per_traj * self.r_n_dir, self.r_dir_dim))
            else:
                self.seq_z_next = None

        self.sample_sequence_z(compute_for_next)

    def sample_sequence_z(self, compute_for_next=False):
        z, z_c, z_d = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
        if self.r_cat_dim > 0:
            gumbel = torch.distributions.Gumbel(ptu.FloatTensor([0]), ptu.FloatTensor([1.0])).sample(self.seq_z_cat.size()).squeeze(-1)
            log_z = torch.log(self.seq_z_cat + eps)
            logit = (log_z + gumbel) / self.temperature
            z = F.softmax(logit, dim=1).view(-1, self.r_n_cat * self.r_cat_dim)
        if self.r_cont_dim > 0:
            normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(self.seq_z_cont_mean.size()).squeeze(-1)
            z_c = self.seq_z_cont_mean + torch.sqrt(self.seq_z_cont_var) * normal
        if self.r_dir_dim > 0:
            if self.r_constraint == 'dirichlet':
                z_d = torch.distributions.Dirichlet(self.seq_z_dir).rsample().view(-1, self.r_n_dir * self.r_dir_dim)
            elif self.r_constraint == 'logitnormal':
                normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(self.seq_z_dir_mean.size()).squeeze(-1)
                z_d = F.softmax(self.seq_z_dir_mean + torch.sqrt(self.seq_z_dir_var) * normal, dim=-1).view(-1, self.r_n_dir * self.r_dir_dim)

        self.seq_z = torch.cat([z, z_c, z_d], dim=-1)

        if compute_for_next:
            z, z_c, z_d = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
            if self.r_cat_dim > 0:
                gumbel = torch.distributions.Gumbel(ptu.FloatTensor([0]), ptu.FloatTensor([1.0])).sample(self.seq_z_next_cat.size()).squeeze(-1)
                log_z = torch.log(self.seq_z_next_cat + eps)
                logit = (log_z + gumbel) / self.temperature
                z = F.softmax(logit, dim=1).view(-1, self.r_n_cat * self.r_cat_dim)
            if self.r_cont_dim > 0:
                normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(self.seq_z_next_cont_mean.size()).squeeze(-1)
                z_c = self.seq_z_next_cont_mean + torch.sqrt(self.seq_z_next_cont_var) * normal
            if self.r_dir_dim > 0:
                if self.r_constraint == 'dirichlet':
                    z_d = torch.distributions.Dirichlet(self.seq_z_next_dir).rsample().view(-1, self.r_n_dir * self.r_dir_dim)
                elif self.r_constraint == 'logitnormal':
                    normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(self.seq_z_next_dir_mean.size()).squeeze(-1)
                    z_d = F.softmax(self.seq_z_next_dir_mean + torch.sqrt(self.seq_z_next_dir_var) * normal, dim=-1).view(-1, self.r_n_dir * self.r_dir_dim)

            self.seq_z_next = torch.cat([z, z_c, z_d], dim=-1)
        else:
            self.seq_z_next = None

    def infer_step_posterior(self, step, resample): 
        num_tasks = 1
        traj_batch = step.shape[0]
        params = self.recurrent_context_encoder(step.view(num_tasks, traj_batch, -1))
        if resample:
            if self.r_constraint == 'logitnormal':
                params = params.view(num_tasks, traj_batch, self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim * 2)
            else:
                params = params.view(num_tasks, traj_batch, self.r_cont_dim * 2 + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim)

            if self.r_cat_dim > 0:
                # params_disc = params[..., :self.r_n_cat * self.r_cat_dim]
                seq_z_cat = params[..., :self.r_n_cat * self.r_cat_dim]
                self.seq_z_cat = F.softmax(seq_z_cat.view(num_tasks * traj_batch * self.r_n_cat, self.r_cat_dim), dim=-1)

            if self.r_cont_dim > 0:
                params_cont = params[..., self.r_n_cat*self.r_cat_dim:self.r_n_cat*self.r_cat_dim+2*self.r_cont_dim]
                seq_z_cont_mean = params_cont[..., :self.r_cont_dim]
                seq_z_cont_var = F.softplus(params_cont[..., self.r_cont_dim:])
                self.seq_z_cont_mean = seq_z_cont_mean.view(num_tasks * traj_batch, self.r_cont_dim)
                self.seq_z_cont_var = seq_z_cont_var.view(num_tasks * traj_batch, self.r_cont_dim)

            if self.r_dir_dim > 0 and self.r_constraint == 'logitnormal':
                params_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
                seq_z_dir_mean = params_dir[..., :self.r_n_dir * self.r_dir_dim]
                seq_z_dir_var = F.softplus(params_dir[..., self.r_dir_dim * self.r_n_dir:])
                self.seq_z_dir_mean = seq_z_dir_mean.view(num_tasks * traj_batch * self.r_n_dir, self.r_dir_dim)
                self.seq_z_dir_var = seq_z_dir_var.view(num_tasks * traj_batch * self.r_n_dir, self.r_dir_dim)

            if self.r_dir_dim > 0 and self.r_constraint == 'dirichlet':
                seq_z_dir = params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
                self.seq_z_dir = F.softplus(seq_z_dir.view(num_tasks * traj_batch * self.r_n_dir, self.r_dir_dim))

            self.sample_sequence_z()

    def clear_sequence_z(self, num_tasks=1, batch_size=1, traj_batch_size=1): 
        assert self.recurrent_context_encoder != None
        if self.r_cat_dim > 0:
            self.seq_z_cat = ptu.ones(num_tasks * batch_size * self.r_n_cat, self.r_cat_dim) / self.r_cat_dim
            self.seq_z_next_cat = None
        if self.r_cont_dim > 0:
            self.seq_z_cont_mean = ptu.zeros(num_tasks * batch_size, self.r_cont_dim)
            self.seq_z_cont_var = ptu.ones(num_tasks * batch_size, self.r_cont_dim)
            self.seq_z_next_cont_mean = None
            self.seq_z_next_cont_var = None
        if self.r_dir_dim > 0:
            if self.r_constraint == 'logitnormal':
                self.seq_z_dir_mean = ptu.zeros(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim)
                self.seq_z_dir_var = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_var
                self.seq_z_next_dir_mean = None
                self.seq_z_next_dir_var = None
            elif self.r_constraint == 'dirichlet':
                self.seq_z_dir = ptu.ones(num_tasks * batch_size * self.r_n_dir, self.r_dir_dim) * self.r_alpha
                self.seq_z_next_dir = None

        self.sample_sequence_z()
        self.recurrent_context_encoder.reset(num_tasks*traj_batch_size)
