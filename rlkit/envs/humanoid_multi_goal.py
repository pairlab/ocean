import numpy as np

from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from . import register_env

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


@register_env('humanoid-multi-goal')
class HumanoidMultiGoalEnv(HumanoidEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, n_dirs=3, radius=4, max_eps=500, seed=0):
        self._max_eps = max_eps
        self._radius = radius
        self._num_steps = 0
        self.tasks = self.sample_tasks(n_tasks, n_dirs)
        self._goals = self.tasks[0]['goal']
        self._goal_steps = self.tasks[0]['step']
        self._goal = self.tasks[0].get('goal', [1])[0]
        super(HumanoidMultiGoalEnv, self).__init__()
        self.seed(seed)

    def step(self, action):
        # pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_reward = -np.sum(np.abs(pos_after - self._goal)) # make it happy, not suicidal
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = goal_reward - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        self._num_steps += 1
        self._goal = self._goals[np.searchsorted(self._goal_steps, self._num_steps)]

        return self._get_obs(), reward, done, dict(reward_linvel=goal_reward,
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
        self._goals = self._task['goal']
        self._goal = self._goals[np.searchsorted(self._goal_steps, self._num_steps)]
        self.reset()

    def sample_tasks(self, num_tasks, num_dirs):
        a = np.random.random((num_tasks, num_dirs)) * 2 * np.pi
        r = self._radius * np.random.random((num_tasks, num_dirs)) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        change_steps = np.sort(np.array([self._max_eps * i / num_dirs for i in range(1, num_dirs)]) + np.random.uniform(-0.05*self._max_eps, 0.05*self._max_eps, size=(num_tasks, num_dirs - 1)))
        # tasks = [{'goal': goal} for goal in goals]
        tasks = []
        for i in range(num_tasks):
            tasks.append({'goal': goals[i], 'step': change_steps[i]})

        return tasks

    def save_all_tasks(self, save_dir):
        import pickle
        import os
        with open(os.path.join(save_dir, 'goals.pkl'), 'wb') as f:
            pickle.dump(self.tasks, f)