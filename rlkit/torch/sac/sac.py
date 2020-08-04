from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import math

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm

import os
import pdb


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            # L[int(i+c*period)] = v
            yield v
            v += step
            i += 1
        yield 1.
    return L 

class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            kl_anneal=False,
            optimizer_class=optim.Adam,
            recurrent=False,
            glob=False,
            use_information_bottleneck=True,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            temp_res=1,
            rnn_sample=None,
            use_next_obs=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.temp_res = temp_res
        self.rnn_sample = rnn_sample

        self.glob = glob
        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda
        self.kl_anneal = kl_anneal

        self.use_next_obs = use_next_obs

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        self.qf1, self.qf2, self.target_qf1, self.target_qf2 = nets[1:]

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        if self.glob:
            self.context_optimizer = optimizer_class(
                self.agent.global_context_encoder.parameters(),
                lr=context_lr,
            )
        if self.recurrent:
            self.recurrent_context_optimizer = optimizer_class(
                self.agent.recurrent_context_encoder.parameters(),
                lr=context_lr,
            )
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()

            self.log_alpha = torch.zeros(1, requires_grad=True, device=ptu.device)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr = policy_lr
            )
        self._n_train_steps_total = 0
        if self.kl_anneal == 'cycle':
            self.annealer = frange_cycle_linear(self.num_iterations * self.num_train_steps_per_itr, n_cycle=20000, ratio=0.7)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.use_automatic_entropy_tuning:
            self.log_alpha.to(device)

    ##### Data handling #####
    def sample_data(self, indices, encoder=False):
        ''' sample data from replay buffers to construct a training meta-batch '''
        # collect data from multiple tasks for the meta-batch
        obs, actions, rewards, next_obs, terms, trajectories, indices_in_trajs = [], [], [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size))
            else:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size, sequence=self.recurrent, traj_batch_size=self.traj_batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            if encoder and self.sparse_rewards:
                # in sparse reward settings, only the encoder is trained with sparse reward
                r = batch['sparse_rewards'][None, ...]
            else:
                r = batch['rewards'][None, ...]
            traj = batch['trajectories'][None, ...]
            indices_in_traj = batch['indices_in_traj'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
            trajectories.append(traj)
            indices_in_trajs.append(indices_in_traj)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        trajectories = torch.cat(trajectories, dim=0)
        indices_in_trajs = torch.cat(indices_in_trajs, dim=0)
        return [obs, actions, rewards, next_obs, terms, trajectories, indices_in_trajs]

    def prepare_encoder_data(self, obs, act, rewards, next_obs):
        ''' prepare context for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        if self.use_next_obs:
            task_data = torch.cat([obs, act, rewards, next_obs], dim=2)
        else:
            task_data = torch.cat([obs, act, rewards], dim=2)
        return task_data

    def prepare_context(self, idx):
        ''' sample context from replay buffer and prepare it '''
        batch = ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size))
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        next_obs = batch['next_observations'][None, ...]
        context = self.prepare_encoder_data(obs, act, rewards, next_obs)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size
        batch = self.sample_data(indices, encoder=True) 
        #! batch = [obs, actions, rewards, ..., terms], each component of batch with shape [indices num, number of steps, obs/action shape]

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices), batch_size=self.batch_size, traj_batch_size=self.traj_batch_size)

        for i in range(num_updates):
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, next_obs_enc, _, _, _ = mini_batch
            if self.glob:
                context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc, next_obs_enc)
            else:
                context = None
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def _take_step(self, indices, context):
        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms, trajectories, indices_in_trajs = self.sample_data(indices)
        indices_in_trajs = indices_in_trajs.long()

        # run inference in networks
        policy_outputs, task_z, seq_z = self.agent(obs, context, trajectories, indices_in_trajs, do_inference=True, compute_for_next=True, is_next=False)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        '''!
        concat task_z with recurrent z if recurrent else orginal task_z
        '''
        q1_pred = self.qf1(obs, actions, torch.cat([task_z, seq_z], dim=-1))
        q2_pred = self.qf2(obs, actions, torch.cat([task_z, seq_z], dim=-1))

        # KL constraint on z if probabilistic
        if self.glob:
            self.context_optimizer.zero_grad()
        if self.recurrent:
            self.recurrent_context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div_disc, kl_div_cont, kl_div_dir, kl_div_seq_disc, kl_div_seq_cont, kl_div_seq_dir = self.agent.compute_kl_div()
            kl_divergence = kl_div_disc + kl_div_cont + kl_div_dir + kl_div_seq_disc + kl_div_seq_cont + kl_div_seq_dir
            if self.kl_anneal == 'mono':
                kl_loss = self.kl_lambda * (kl_div_disc + kl_div_cont + kl_div_dir + kl_div_seq_disc + kl_div_seq_cont + kl_div_seq_dir) * (1 - math.exp(-(self._n_train_steps_total+100) / 100000.0))
            elif self.kl_anneal == 'cycle':
                kl_loss = self.kl_lambda * (kl_div_disc + kl_div_cont + kl_div_dir + kl_div_seq_disc + kl_div_seq_cont + kl_div_seq_dir) * float(next(self.annealer))
            elif self.kl_anneal == 'none':
                kl_loss = self.kl_lambda * (kl_div_disc + kl_div_cont + kl_div_dir + kl_div_seq_disc + kl_div_seq_cont + kl_div_seq_dir)
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy)
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)

        # calculate q target
        next_policy_outputs, _, seq_z_next = self.agent(next_obs, context, trajectories, indices_in_trajs, do_inference=False, compute_for_next=False, is_next=True)
        new_next_actions, _, _, new_log_pi = next_policy_outputs[:4]
        next_obs = next_obs.view(t * b, -1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions, torch.cat([task_z, seq_z_next], dim=-1)),
            self.target_qf2(next_obs, new_next_actions, torch.cat([task_z, seq_z_next], dim=-1)),
        ) - alpha * new_log_pi
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        # compute min Q on the new actions
        q_new_actions = torch.min(
            self.qf1(obs, new_actions, torch.cat([task_z.detach(), seq_z_next.detach()], dim=-1)),
            self.qf2(obs, new_actions, torch.cat([task_z.detach(), seq_z_next.detach()], dim=-1)),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        if self.glob:
            self.context_optimizer.step()
        if self.recurrent:
            self.recurrent_context_optimizer.step()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self._update_target_network()

        # save some statistics for eval
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                if self.agent.g_cat_dim > 0:
                    z_mean = ptu.get_numpy(self.agent.z_means[0])
                    for i in range(len(z_mean)):
                        self.eval_statistics['Z mean disc train %d'%i] = z_mean[i]
                if self.agent.g_cont_dim > 0:
                    z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_c_means[0])))
                    z_sig = np.mean(ptu.get_numpy(self.agent.z_c_vars[0]))
                    self.eval_statistics['Z mean cont train'] = z_mean
                    self.eval_statistics['Z variance cont train'] = z_sig
                if self.agent.g_dir_dim > 0 and self.agent.g_constraint == 'logitnormal':
                    z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_d_means[0])))
                    z_sig = np.mean(ptu.get_numpy(self.agent.z_d_vars[0]))
                    self.eval_statistics['Z mean dir train'] = z_mean
                    self.eval_statistics['Z variance dir train'] = z_sig
                self.eval_statistics['KL Cont Divergence'] = ptu.get_numpy(kl_div_cont)
                self.eval_statistics['KL Disc Divergence'] = ptu.get_numpy(kl_div_disc)
                self.eval_statistics['KL Dir Divergence'] = ptu.get_numpy(kl_div_dir)
                self.eval_statistics['KL RNN Cont Divergence'] = ptu.get_numpy(kl_div_seq_cont)
                self.eval_statistics['KL RNN Disc Divergence'] = ptu.get_numpy(kl_div_seq_disc)
                self.eval_statistics['KL RNN Dir Divergence'] = ptu.get_numpy(kl_div_seq_dir)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
                self.eval_statistics['KL Div'] = ptu.get_numpy(kl_divergence)

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            target_qf1=self.target_qf1.state_dict(),
            target_qf2=self.target_qf2.state_dict(),
        )
        if self.recurrent:
            snapshot['recurrent_context_encoder'] = self.agent.recurrent_context_encoder.state_dict()
        if self.glob:
            snapshot['global_context_encoder'] = self.agent.global_context_encoder.state_dict()
        return snapshot
