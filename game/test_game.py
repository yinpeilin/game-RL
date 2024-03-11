import gymnasium as gym
import numpy as np
import csv
from copy import deepcopy

class CartPoleEnv():
    def __init__(self, worker_id, frame_tick = 1, num_tick = 1, monitor_file_path = None):
        self.game = gym.make('CartPole-v1', render_mode="rgb_array").unwrapped
        self.num_tick = num_tick
        self.frame_tick = frame_tick
        self.action_num = self.game.action_space.n
        self.state = {"box_observation": np.zeros((self.num_tick,4), dtype=np.float32)}
        
        self.monitor_file_path = monitor_file_path
        self.ticks:int = 0
        self.all_reward = 0.0
    def reset(self, seed = None):
        state, info = self.game.reset()
        
        for key in self.state.keys():
            if key == "box_observation":
                self.state[key] =  np.zeros((self.num_tick,4), dtype=np.float32)
                self.state[key][-1] = np.array(state, dtype=np.float32)
        
        if self.monitor_file_path != None:
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
            self.ticks += 1
            if done == True:
                step_reward = -10
                break
            i+=1
        self.all_reward += save_reward
        
        for key in self.state.keys():
            if key == "box_observation":
                self.state[key][0:-1] = self.state[key][1:]
                self.state[key][-1] = np.array(state, dtype=np.float32)
        
        return deepcopy(self.state), step_reward, done, truncated, info
    def reset_step(self, action_index):
        return_state, step_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()
        return return_state, step_reward, done, truncated, info
    def render(self):
        image = self.game.render()
        import cv2
        cv2.imshow("render_image", image)
        cv2.waitKey(10)