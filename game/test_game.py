import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv
from copy import deepcopy

class CartPoleEnv(gym.Env):
    def __init__(self, frame_tick = 1, num_tick = 4, monitor_file_path = 'test.csv'):
        self.game = gym.make('CartPole-v1', render_mode="human")
        self.num_tick = num_tick
        self.frame_tick = frame_tick
        
        self.action_space = self.game.action_space
        
        self.observation_space = spaces.Dict({
            "box_observation": self.game.observation_space,
        })
        
        self.state = {"box_observation": np.zeros((self.num_tick,4), dtype=np.float32)}
        
        self.monitor_file= csv.writer(open(monitor_file_path, 'a', newline=''))
        self.ticks:int = 0
        self.all_reward = 0.0
        
    def reset(self, seed = None):
        state, info = self.game.reset()
        self.state["box_observation"] =  np.zeros((self.num_tick,4), dtype=np.float32)
        self.state["box_observation"][-1] = np.array(state, dtype=np.float32)
        self.monitor_file.writerow([int(self.ticks), int(self.all_reward)])
        self.ticks = 0
        self.all_reward = 0.0
        
        return self.state, info
    
    def step(self, action_index):
        all_reward: float = 0.0
        i: int = 0
        while i < self.frame_tick:
            state, reward, done, truncated, info = self.game.step(action_index)
            all_reward += reward
            self.ticks += 1
            
            if done == True:
                break
            i+=1
            
        self.all_reward += all_reward
        
        self.state["box_observation"][0:-1] = self.state["box_observation"][1:]
        self.state["box_observation"][-1] = np.array(state, dtype=np.float32)/255.0
        
        return self.state, all_reward, done, truncated, info
    
    def reset_step(self, action_index):
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            return_state = deepcopy(self.state)
            self.reset()
        return return_state, all_reward, done, truncated, info
        
    def render(self):
        self.game.render()
    