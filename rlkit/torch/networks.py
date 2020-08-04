"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm

import numpy as np

def identity(x):
    return x

def read_dim(s):
    a, b, c, d, e = s.split('.')
    return [int(a), int(b), int(c), int(d), int(e)]

eps = 1e-11

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            set_activation=torch.sum,
            set_output_size=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.set_activation = set_activation
        self.set_output_size = set_output_size
        self.b_init_value = b_init_value
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)

class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)

class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass

class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hn', torch.zeros(1, 1, self.hidden_dim))
        self.register_buffer('cn', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (1, task, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            if self.layer_norm and i < len(self.fcs) - 1:
                out = self.layer_norms[i](out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hn, self.cn))
        self.hn = hn
        self.cn = cn
        # take the last hidden state to predict z
        # out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hn = self.hn.new_full((1, num_tasks, self.hidden_dim), 0)
        self.cn = self.cn.new_full((1, num_tasks, self.hidden_dim), 0)

class VRNNEncoder(PyTorchModule):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                hidden_sizes,
                output_size,
                input_size,
                temperature,
                vrnn_latent,
                vrnn_constraint,
                r_alpha,
                r_var,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.temperature = temperature
        self.r_cont_dim, self.r_n_cat, self.r_cat_dim, self.r_n_dir, self.r_dir_dim = read_dim(vrnn_latent)
        self.z_size = self.r_cont_dim + self.r_n_cat * self.r_cat_dim + self.r_n_dir * self.r_dir_dim
        self.vrnn_constraint = vrnn_constraint
        self.r_alpha = r_alpha
        self.r_var = r_var
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hn', torch.zeros(1, self.hidden_dim))
        self.register_buffer('cn', torch.zeros(1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (1, task, feat)

        self.rnn = nn.GRUCell(self.hidden_dim * 2, self.hidden_dim)
        self.prior = Mlp(hidden_sizes=[self.hidden_dim], input_size=self.hidden_dim, output_size=self.output_size)
        self.phi_z = Mlp(hidden_sizes=[self.hidden_dim], input_size=self.z_size, output_size=self.hidden_dim)
        self.phi_x = Mlp(hidden_sizes=[self.hidden_dim], input_size=self.input_size, output_size=self.hidden_dim)
        self.encoder = Mlp(hidden_sizes=[self.hidden_dim], input_size=self.hidden_dim * 2, output_size=self.output_size)

        if self.r_cat_dim > 0:
            self.z_cat_prior_dist = torch.distributions.Categorical(ptu.ones(self.r_cat_dim) / self.r_cat_dim)
        if self.r_dir_dim > 0:
            if self.vrnn_constraint == 'logitnormal':
                self.z_dir_prior_dist = torch.distributions.Normal(ptu.zeros(self.r_dir_dim), ptu.ones(self.r_dir_dim)*np.sqrt(self.r_var))
            elif self.vrnn_constraint == 'dirichlet':
                self.z_dir_prior_dist = torch.distributions.Dirichlet(ptu.ones(self.r_dir_dim) * self.r_alpha)
        if self.r_cont_dim > 0:
            self.z_cont_prior_dist = torch.distributions.Normal(ptu.zeros(self.r_cont_dim), ptu.ones(self.r_cont_dim))

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        assert not return_preactivations
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        out = self.phi_x(out)
        out = out.view(task, seq, self.hidden_dim)
        # out = out.transpose(0, 1)

        output = []
        self.kl_div_seq_cont, self.kl_div_seq_disc, self.kl_div_seq_dir = ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean()
        for i in range(seq):
            # phi_x_i = out[i]
            phi_x_i = out[:, i, :]

            enc_i = self.encoder(torch.cat([phi_x_i, self.hn], dim = -1))
            output.append(enc_i)
            
            prior_i = self.prior(self.hn)
            z_i, kl_disc, kl_cont, kl_dir = self.achieve_params_rsample_kl(enc_i, prior_i)

            phi_z_i = self.phi_z(z_i)

            self.hn = self.rnn(torch.cat([phi_x_i, phi_z_i], dim = -1), self.hn)

            self.kl_div_seq_disc += kl_disc
            self.kl_div_seq_cont += kl_cont
            self.kl_div_seq_dir += kl_dir

        output = torch.stack(output)
        # print (output.is_contiguous())
        output = output.transpose(0, 1)
        # print (output.is_contiguous())
        # exit(-1)
        return output

    def reset(self, num_tasks=1):
        self.hn = self.hn.new_full((num_tasks, self.hidden_dim), 0)
        self.cn = self.cn.new_full((num_tasks, self.hidden_dim), 0)

    def compute_kl_div(self):
        return self.kl_div_seq_disc, self.kl_div_seq_cont, self.kl_div_seq_dir

    def achieve_params_rsample_kl(self, post_params, prior_params): 
        num_tasks = post_params.size(0)

        z, z_c, z_d = ptu.FloatTensor(), ptu.FloatTensor(), ptu.FloatTensor()
        kl_div_seq_cont, kl_div_seq_disc, kl_div_seq_dir = ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean(), ptu.FloatTensor([0.]).mean()
        if self.r_cat_dim > 0:
            seq_z_cat_post = post_params[..., :self.r_n_cat * self.r_cat_dim]
            seq_z_cat_post = F.softmax(seq_z_cat_post.reshape(num_tasks * self.r_n_cat, self.r_cat_dim), dim=-1)
            posterior = torch.distributions.Categorical(seq_z_cat_post)

            seq_z_cat_prior = prior_params[..., :self.r_n_cat * self.r_cat_dim]
            seq_z_cat_prior = F.softmax(seq_z_cat_prior.reshape(num_tasks * self.r_n_cat, self.r_cat_dim), dim=-1)
            prior = torch.distributions.Categorical(seq_z_cat_prior)
            
            kl_div_seq_disc = torch.sum(torch.distributions.kl.kl_divergence(posterior, prior))

            gumbel = torch.distributions.Gumbel(ptu.FloatTensor([0]), ptu.FloatTensor([1.0])).sample(seq_z_cat_post.size()).squeeze(-1)
            log_z = torch.log(seq_z_cat_post + eps)
            logit = (log_z + gumbel) / self.temperature
            z = F.softmax(logit, dim=1).view(num_tasks, self.r_n_cat * self.r_cat_dim)

        if self.r_cont_dim > 0:
            post_params_cont = post_params[..., self.r_n_cat*self.r_cat_dim:self.r_n_cat*self.r_cat_dim+2*self.r_cont_dim]
            seq_z_cont_mean_post = post_params_cont[..., :self.r_cont_dim]
            seq_z_cont_var_post = F.softplus(post_params_cont[..., self.r_cont_dim:])
            seq_z_cont_mean_post = seq_z_cont_mean_post.reshape(num_tasks, self.r_cont_dim)
            seq_z_cont_var_post = seq_z_cont_var_post.reshape(num_tasks, self.r_cont_dim)
            posterior = torch.distributions.Normal(seq_z_cont_mean_post, torch.sqrt(seq_z_cont_var_post))

            prior_params_cont = prior_params[..., self.r_n_cat*self.r_cat_dim:self.r_n_cat*self.r_cat_dim+2*self.r_cont_dim]
            seq_z_cont_mean_prior = prior_params_cont[..., :self.r_cont_dim]
            seq_z_cont_var_prior = F.softplus(prior_params_cont[..., self.r_cont_dim:])
            seq_z_cont_mean_prior = seq_z_cont_mean_prior.reshape(num_tasks, self.r_cont_dim)
            seq_z_cont_var_prior = seq_z_cont_var_prior.reshape(num_tasks, self.r_cont_dim)
            prior = torch.distributions.Normal(seq_z_cont_mean_prior, torch.sqrt(seq_z_cont_var_prior))
            
            kl_div_seq_cont = torch.sum(torch.distributions.kl.kl_divergence(posterior, prior))
            
            normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(seq_z_cont_mean_post.size()).squeeze(-1)
            z_c = seq_z_cont_mean_post + torch.sqrt(seq_z_cont_var_post) * normal

        if self.r_dir_dim > 0 and self.vrnn_constraint == 'logitnormal':
            post_params_dir = post_params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            seq_z_dir_mean_post = post_params_dir[..., :self.r_n_dir * self.r_dir_dim]
            seq_z_dir_var_post = F.softplus(post_params_dir[..., self.r_dir_dim * self.r_n_dir:])
            seq_z_dir_mean_post = seq_z_dir_mean_post.reshape(num_tasks * self.r_n_dir, self.r_dir_dim)
            seq_z_dir_var_post = seq_z_dir_var_post.reshape(num_tasks * self.r_n_dir, self.r_dir_dim)
            posterior = torch.distributions.Normal(seq_z_dir_mean_post, torch.sqrt(seq_z_dir_var_post))

            prior_params_dir = prior_params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            seq_z_dir_mean_prior = prior_params_dir[..., :self.r_n_dir * self.r_dir_dim]
            seq_z_dir_var_prior = F.softplus(prior_params_dir[..., self.r_dir_dim * self.r_n_dir:])
            seq_z_dir_mean_prior = seq_z_dir_mean_prior.reshape(num_tasks * self.r_n_dir, self.r_dir_dim)
            seq_z_dir_var_prior = seq_z_dir_var_prior.reshape(num_tasks * self.r_n_dir, self.r_dir_dim)
            prior = torch.distributions.Normal(seq_z_dir_mean_prior, torch.sqrt(seq_z_dir_var_prior))
            
            kl_div_seq_dir = torch.sum(torch.distributions.kl.kl_divergence(posterior, prior))

            normal = torch.distributions.Normal(ptu.FloatTensor([0.]), ptu.FloatTensor([1.])).sample(seq_z_dir_mean_post.size()).squeeze(-1)
            z_d = F.softmax(seq_z_dir_mean_post + torch.sqrt(seq_z_dir_var_post) * normal, dim=-1).view(num_tasks, self.r_n_dir * self.r_dir_dim)
        
        if self.r_dir_dim > 0 and self.vrnn_constraint == 'dirichlet':
            seq_z_dir_post = post_params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            seq_z_dir_post = F.softplus(seq_z_dir_post.reshape(num_tasks * self.r_n_dir, self.r_dir_dim))
            posterior = torch.distributions.Dirichlet(seq_z_dir_post)

            seq_z_dir_prior = prior_params[..., self.r_n_cat * self.r_cat_dim + self.r_cont_dim * 2:]
            seq_z_dir_prior = F.softplus(seq_z_dir_prior.reshape(num_tasks * self.r_n_dir, self.r_dir_dim))
            prior = torch.distributions.Dirichlet(seq_z_dir_prior)
            
            kl_div_seq_dir = torch.sum(torch.distributions.kl.kl_divergence(posterior, prior))
            
            z_d = posterior.rsample().view(num_tasks, self.r_n_dir * self.r_dir_dim)

        return torch.cat([z, z_c, z_d], dim=-1), kl_div_seq_disc, kl_div_seq_cont, kl_div_seq_dir
