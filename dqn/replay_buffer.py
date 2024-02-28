import torch as th
from copy import deepcopy

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, buffer_size, obs_shape_dict):
        self.buffer_size = buffer_size
        self.size = 0
        self.index = 0
        self.obs_shape_dict = deepcopy(obs_shape_dict)
        self.states = {}
        self.next_states = {}
        for key in self.obs_shape_dict:
            temp_size = [self.buffer_size]
            temp_size.extend(self.obs_shape_dict[key])
            self.states[key] = th.zeros(temp_size, dtype=th.float32)
            self.next_states[key] = th.zeros(temp_size, dtype=th.float32)
        self.actions = th.zeros((self.buffer_size, ), dtype=th.int32)
        self.rewards = th.zeros((self.buffer_size, ), dtype=th.float32)
        self.dones = th.zeros((self.buffer_size, ), dtype=th.float32)
        self.truncateds = th.zeros((self.buffer_size, ), dtype=th.float32)
        self.weights = th.ones((self.buffer_size, ), dtype=th.float32)

    def add(self, states, actions: th.Tensor, rewards: th.Tensor, next_states, dones: th.Tensor, truncateds: th.Tensor):  # 将数据加入buffer
        end_index = self.buffer_size if (
            self.index + actions.shape[0]) > self.buffer_size else (self.index + actions.shape[0])
        element_size = end_index - self.index
        for key in self.obs_shape_dict:
            self.states[key][self.index:end_index] = states[key][0:element_size]
            self.next_states[key][self.index:end_index] = next_states[key][0:element_size]
        self.actions[self.index:end_index] = actions[0:element_size]
        self.rewards[self.index:end_index] = rewards[0:element_size]
        self.dones[self.index:end_index] = dones[0:element_size]
        self.truncateds[self.index:end_index] = truncateds[0:element_size]
        self.weights[self.index:end_index] = 1.0
        self.size += actions.shape[0]
        if self.size >= self.buffer_size:
            self.size = self.buffer_size
        self.index = end_index % self.buffer_size

    def sample(self, batch_size):

        assert self.size > batch_size, 'sample size can not be larger than buffer size'

        random_indices = th.randint(0, self.size, size=(batch_size,))
        states = {}
        next_states = {}
        for key in self.obs_shape_dict:
            states[key] = self.states[key][random_indices]
            next_states[key] = self.states[key][random_indices]
        actions = self.actions[random_indices]
        rewards = self.rewards[random_indices]
        dones = self.dones[random_indices]
        truncateds =  self.truncateds[random_indices]
        
        return states, actions, rewards, next_states, dones, truncateds
    
    def update_priorities(self, indices, prios):
        max_value = prios.max() + 1e-5
        min_value = prios.min()
        self.weights[indices]*=0.5
        self.weights[indices]+= 10*(prios - min_value)/(max_value - min_value)
