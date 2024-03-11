import gymnasium as gym
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import cv2
import csv
from copy import deepcopy

class mario_env():
    def __init__(self, worker_id, frame_tick = 4, num_tick = 4, monitor_file_path = None):
        
        self.game = JoypadSpace(gym.make('SuperMarioBros-1-1-v0'), COMPLEX_MOVEMENT)

        self.num_tick = num_tick
        self.frame_tick = frame_tick
        
        self.state = {
            'image': np.zeros((self.num_tick, 100, 100), dtype=np.float32),
            "tick": np.zeros((self.num_tick, 1), dtype = np.float32),
            "last_press": np.zeros((self.num_tick, len(COMPLEX_MOVEMENT)), dtype = np.int32)   
        }
        
        self.action_num = len(COMPLEX_MOVEMENT)
        self.monitor_file_path = monitor_file_path
        self.ticks = 0.0
        self.all_reward = 0.0
    def reset(self, seed = None):
        state, info = self.game.reset()
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (100, 100))
        
        
        with open(self.monitor_file_path, 'a', newline='') as fp:
            monitor_file = csv.writer(fp)
            monitor_file.writerow([int(self.ticks), float(self.all_reward)])
        self.ticks = 0.0
        
        self.state['image'] = np.zeros((self.num_tick, 100, 100), dtype=np.float32)
        self.state['image'][-1] = np.array(state, dtype=np.float32)/255.0
        self.state['tick'] = np.zeros((self.num_tick, 1), dtype = np.float32)
        self.state["last_press"] = np.zeros((self.num_tick, len(COMPLEX_MOVEMENT)), dtype = np.float32)
        
        self.all_reward = 0.0
        
        return deepcopy(self.state), info
    
    def step(self, action_index):
        
        all_reward = 0.0
        for i in range(self.frame_tick):
            state, reward, done, truncated, info = self.game.step(action_index)
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            state = cv2.resize(state, (100, 100))
            all_reward += reward
            self.ticks += 1.0
            if self.ticks >= 3000:
                truncated = True
                done = True
            if done == True:
                break
        self.all_reward += all_reward
        
        self.state['image'][0:-1] = self.state['image'][1:]
        self.state['image'][-1] = np.array(state, dtype=np.float32)/255.0
        self.state['tick'][0:-1] = self.state['tick'][1:]
        self.state['tick'][-1] = self.ticks/2000
        self.state["last_press"][0:-1] = self.state["last_press"][1:]
        self.state["last_press"][-1, :] = 0.0
        self.state["last_press"][-1, action_index] = 1.0
        
        return deepcopy(self.state), all_reward, done, truncated, info
    
    def reset_step(self, action_index):
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()
        return return_state, all_reward, done, truncated, info
        
    def render(self):
        import cv2
        cv2.imshow("render_image", self.state['image'][-1])
        cv2.waitKey(10)

def make_mario_env():
    
    return mario_env()

if __name__ == '__main__':
    env = mario_env(frame_tick=4)
    done = True
    
    frame = 0
    while True:
        if done == True:
            env.reset()
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)
        print(frame)
        # state, reward, done, truncated, info = env.step(env.action_space.sample())
        frame +=1
        env.render()
