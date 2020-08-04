import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer

def zero_padding(arr, length):
    return np.pad(arr, ((0,length-len(arr)),(0,0)), 'constant')

class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim, max_episode_len,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._max_episode_len = int(max_episode_len)
        self._observations = np.zeros((max_replay_buffer_size, max_episode_len, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, max_episode_len, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, max_episode_len, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, max_episode_len, 1))
        self._sparse_rewards = np.zeros((max_replay_buffer_size, max_episode_len, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, max_episode_len, 1), dtype='uint8')
        self.clear()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = zero_padding(observation, self._max_episode_len)
        self._actions[self._top] = zero_padding(action, self._max_episode_len)
        self._rewards[self._top] = zero_padding(reward, self._max_episode_len)
        self._terminals[self._top] = zero_padding(terminal, self._max_episode_len)
        self._next_obs[self._top] = zero_padding(next_observation, self._max_episode_len)
        if 'sparse_reward' in kwargs['env_info'][0]:
            self._sparse_rewards[self._top] = zero_padding(np.array([[d['sparse_reward']] for d in kwargs['env_info']]), 
                                                                self._max_episode_len)
        self._episode_len[self._top] = len(observation)
        self._advance()
        self._range = []
        for i in range(self._size):
            self._range.extend(list(range(i*self._max_episode_len, i*self._max_episode_len+int(self._episode_len[i]))))

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        pass

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_len = np.zeros(self._max_replay_buffer_size)
        self._range = []

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices, traj_indices=None, indices_in_traj=None): 
        if type(traj_indices) != type(None):
            trajectories = np.concatenate([self._observations[traj_indices], self._actions[traj_indices], self._rewards[traj_indices]], axis=-1)
            indices_in_traj = np.array(indices_in_traj)
        else:
            trajectories = np.zeros((len(indices), 1)) # placeholder, shape should be [traj_batch_size or len(traj_indices), obs_dim + action_dim + rew_dim]
            indices_in_traj = np.zeros((len(indices), 1)) # shape should be [traj batch size, batch size//traj batch size]
        return dict(
            observations=self._observations.reshape(self._max_replay_buffer_size * self._max_episode_len, -1)[indices],
            actions=self._actions.reshape(self._max_replay_buffer_size * self._max_episode_len, -1)[indices],
            rewards=self._rewards.reshape(self._max_replay_buffer_size * self._max_episode_len, -1)[indices],
            terminals=self._terminals.reshape(self._max_replay_buffer_size * self._max_episode_len, -1)[indices],
            next_observations=self._next_obs.reshape(self._max_replay_buffer_size * self._max_episode_len, -1)[indices],
            sparse_rewards=self._sparse_rewards.reshape(self._max_replay_buffer_size * self._max_episode_len, -1)[indices],
            trajectories=trajectories,
            indices_in_traj=indices_in_traj,
        )

    def random_batch(self, batch_size, sequence=False, traj_batch_size=-1): 
        ''' batch of unordered transitions '''
        # indices = np.random.randint(0, self._size, batch_size)
        if traj_batch_size == -1:
            assert sequence == False
            indices = np.random.choice(self._range, batch_size)
            traj_indices = None
            indices_in_traj = None
        else:
            traj_indices = np.random.choice(self._size, traj_batch_size)
            indices, indices_in_traj = [], []

            for i in traj_indices:
                batch_traj_indices = np.random.choice(range(int(self._episode_len[i])), batch_size//traj_batch_size)
                indices.extend(list(batch_traj_indices+i*self._max_episode_len))
                indices_in_traj.append(list(batch_traj_indices))

        return self.sample_data(indices, traj_indices, indices_in_traj)

    def num_steps_can_sample(self):
        return self._size

    def return_save_data(self):
        return {'obs': self._observations, 'next_obs': self._next_obs, 'actions': self._actions, 'rewards': self._rewards, \
                'sparse_rewards': self._sparse_rewards, 'terminals': self._terminals, 'top': self._top, 'eps_len': self._episode_len, \
                'range': self._range, 'size': self._size}

    def load_save_data(self, d):
        self._observations = d['obs']
        self._next_obs = d['next_obs']
        self._actions = d['actions']
        self._rewards = d['rewards']
        self._sparse_rewards = d['sparse_rewards']
        self._terminals = d['terminals']
        self._top = d['top']
        self._episode_len = d['_episode_len']
        self._range = d['range']
        self._size = d['size']