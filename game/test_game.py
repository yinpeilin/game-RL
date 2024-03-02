import gymnasium as gym
from gymnasium import spaces
import numpy as np
import csv
from copy import deepcopy

class CartPoleEnv(gym.Env):
    def __init__(self, frame_tick = 1, num_tick = 1, monitor_file_path = 'test.csv'):
        self.game = gym.make('CartPole-v1', render_mode="human").unwrapped
        self.num_tick = num_tick
        self.frame_tick = frame_tick
        self.action_space = self.game.action_space
        self.observation_space = spaces.Dict({
            "box_observation": self.game.observation_space,
        })
        self.state = {"box_observation": np.zeros((1,self.num_tick,4), dtype=np.float32)}
        self.monitor_file_path = monitor_file_path
        self.ticks:int = 0
        self.all_reward = 0.0
    def reset(self, seed = None):
        state, info = self.game.reset()
        self.state["box_observation"] =  np.zeros((1, self.num_tick,4), dtype=np.float32)
        self.state["box_observation"][0, -1] = np.array(state, dtype=np.float32)
        
        with open(self.monitor_file_path, 'a', newline='') as fp:
            monitor_file = csv.writer(fp)
            monitor_file.writerow([self.ticks, self.all_reward])
        self.ticks = 0
        self.all_reward = 0.0
        return deepcopy(self.state), info
    def step(self, action_index):
        save_reward: float = 0.0
        step_reward: float = 0.0
        i: int = 0
        while i < self.frame_tick:
            state, reward, done, truncated, info = self.game.step(action_index)
            save_reward += reward
            x, __, theta, __ = state
            r1 = (self.game.x_threshold - abs(x)) / self.game.x_threshold - 0.8
            r2 = (self.game.theta_threshold_radians - abs(theta)) / self.game.theta_threshold_radians - 0.5
            step_reward += (r1+r2)
            # step_reward = reward
            self.ticks += 1
            if done == True:
                step_reward = -10
                break
            i+=1
        self.all_reward += save_reward
        self.state["box_observation"][0, 0:-1] = self.state["box_observation"][0, 1:]
        self.state["box_observation"][0, -1] = np.array(state, dtype=np.float32)
        return deepcopy(self.state), step_reward, done, truncated, info
    def reset_step(self, action_index):
        return_state, step_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()
        return return_state, step_reward, done, truncated, info
    def render(self):
        self.game.render()