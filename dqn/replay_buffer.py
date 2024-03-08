import torch as th
import numpy as np
from copy import deepcopy
class ReplayBuffer(object):
    def __init__(self, buffer_size, obs_shape_dict):
        self.buffer_size = buffer_size
        self.size = 0
        self.index = 0
        self.obs_shape_dict = deepcopy(obs_shape_dict)
        
        self.states = {}
        self.next_states = {}
        
        for key, value in self.obs_shape_dict.items():
            state_shape = [self.buffer_size]
            state_shape.extend(value)
            self.states[key] = np.zeros(state_shape, dtype = np.float32)
            self.next_states[key] = np.zeros(state_shape, dtype = np.float32)
        # self.states = np.zeros((self.buffer_size, 4), dtype=np.float32)
        # self.next_states = np.zeros((self.buffer_size, 4), dtype=np.float32)
        
        self.actions = np.zeros((self.buffer_size, ), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, ), dtype=np.float32)
        self.truncateds = np.zeros((self.buffer_size, ), dtype=np.float32)
        # self.weights = np.ones((self.buffer_size, ), dtype=np.float32)

    def add(self, states, actions, rewards, next_states, dones, truncateds):  # 将数据加入buffer
        end_index = self.buffer_size if (
            self.index + actions.shape[0]) > self.buffer_size else (self.index + actions.shape[0])
        element_size = end_index - self.index
        
        for key in self.obs_shape_dict.keys():
            self.states[key][self.index:end_index] = states[key][0:element_size]
            self.next_states[key][self.index:end_index] = next_states[key][0:element_size]
        self.actions[self.index:end_index] = actions[0:element_size]
        self.rewards[self.index:end_index] = rewards[0:element_size]
        self.dones[self.index:end_index] = dones[0:element_size]
        self.truncateds[self.index:end_index] = truncateds[0:element_size]
        # self.weights[self.index:end_index] = 1.0
        
        self.size += actions.shape[0]
        self.size = self.buffer_size if self.size >= self.buffer_size else self.size
        self.index = end_index % self.buffer_size
    def sample(self, batch_size):
        assert self.size > batch_size, 'sample size can not be larger than buffer size'
        random_indices = np.random.randint(0, self.size, size=batch_size)
        states = {}
        next_states = {}
        for key in self.obs_shape_dict.keys():
            states[key] =  th.FloatTensor(self.states[key][random_indices])
            next_states[key] =  th.FloatTensor(self.next_states[key][random_indices])
        actions = th.LongTensor(self.actions[random_indices])
        rewards = th.FloatTensor(self.rewards[random_indices])
        dones = th.FloatTensor(self.dones[random_indices])
        truncateds =  th.FloatTensor(self.truncateds[random_indices])
        # weights = th.FloatTensor(self.weights[random_indices])
        return states, actions, rewards, next_states, dones, truncateds 
    # weights, random_indices
    
    # def update_priorities(self, indices, prios):
    #     max_value = prios.max() + 1e-5
    #     min_value = prios.min()
    #     self.weights[indices]*=0.9
    #     self.weights[indices]+= 1*(prios - min_value)/(max_value - min_value)
