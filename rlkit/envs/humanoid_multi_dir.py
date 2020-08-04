import numpy as np

from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from . import register_env

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


@register_env('humanoid-multi-dir')
class HumanoidMultiDirEnv(HumanoidEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, n_dirs=3, max_eps=700, seed=0):
        self._max_eps = max_eps
        self._num_steps = 0
        self.tasks = self.sample_tasks(n_tasks, n_dirs)
        self._goal_dirs = self.tasks[0]['dir']
        self._goal_steps = self.tasks[0]['step']
        self._goal_dir = self.tasks[0].get('dir', [1])[0]
        self._goal = self._goal_dir
        super(HumanoidMultiDirEnv, self).__init__()
        self.seed(seed)

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self._goal_dir), np.sin(self._goal_dir))
        lin_vel_cost = np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        self._num_steps += 1
        self._goal_dir = self._goal_dirs[np.searchsorted(self._goal_steps, self._num_steps)]

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._num_steps = 0
        self._goal_steps = self._task['step']
        self._goal_dirs = self._task['dir']
        self._goal_dir = self._goal_dirs[np.searchsorted(self._goal_steps, self._num_steps)]
        self._goal = self._goal_dir
        self.reset()

    def sample_tasks(self, num_tasks, num_dirs):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks, num_dirs))
        change_steps = np.sort(np.array([self._max_eps * i / num_dirs for i in range(1, num_dirs)]) + np.random.uniform(-0.05*self._max_eps, 0.05*self._max_eps, size=(num_tasks, num_dirs - 1)))
        tasks = []
        for i in range(num_tasks):
            tasks.append({'dir': directions[i], 'step': change_steps[i]})
        return tasks

    def save_all_tasks(self, save_dir):
        import pickle
        import os
        with open(os.path.join(save_dir, 'goals.pkl'), 'wb') as f:
            pickle.dump(self.tasks, f)