import gymnasium as gym
from gymnasium import spaces
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import cv2
import csv
from copy import deepcopy

global_world_index_list = [1, 2, 3, 4, 5, 6, 7, 8]
global_stage_index_list = [1, 2, 3, 4]
version_index_list = [0, 1, 2, 3]
class mario_env(gym.Env):
    def __init__(self, world_index_list = None, stage_index_list = None, version_index = None, frame_tick = 4, num_tick = 4, monitor_file_path = 'test.csv'):
        if world_index_list == None:
            world_index_list = global_world_index_list
        if stage_index_list == None:
            stage_index_list = global_stage_index_list
        if version_index == None:
            version_index = 0
            
        str_temp_list = []
        for i in world_index_list:
            for j in stage_index_list:
                str_temp_list.append('{}-{}'.format(i,j))
                
        # game = gym.make('SuperMarioBrosRandomStages-v0', stages= str_temp_list)
        game = gym.make('SuperMarioBros-1-1-v0')
        self.game = JoypadSpace(game, COMPLEX_MOVEMENT)

        self.num_tick = num_tick
        self.frame_tick = frame_tick
        
        self.state = {
            'image': np.zeros((1, self.num_tick, 100, 100), dtype=np.float32),
            "tick": np.zeros((1, self.num_tick, 1), dtype = np.float32),
            "last_press": np.zeros((1, self.num_tick, len(COMPLEX_MOVEMENT)), dtype = np.int32)   
        }
        
        self.action_space = spaces.Discrete(len(COMPLEX_MOVEMENT))
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(self.num_tick, 100, 100), dtype=np.float32),
            "tick": spaces.Box(low=0, high=1, shape=(self.num_tick, 1), dtype=np.float32),
            "last_press":spaces.Box(low=0, high=1, shape=(self.num_tick,len(COMPLEX_MOVEMENT)), dtype=np.float32)
        })
        self.monitor_file_path = monitor_file_path
        self.ticks = 0.0
        self.all_reward = 0.0
    def reset(self, seed = None):
        state, info = self.game.reset()
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (100, 100))
        
        self.state['image'] = np.zeros((1, self.num_tick, 100, 100), dtype=np.float32)
        self.state['image'][0, -1] = np.array(state, dtype=np.float32)/255.0
        with open(self.monitor_file_path, 'a', newline='') as fp:
            monitor_file = csv.writer(fp)
            monitor_file.writerow([int(self.ticks), float(self.all_reward)])
        self.ticks = 0.0
        
        self.state['tick'] = np.zeros((1, self.num_tick, 1), dtype = np.float32)
        self.state["last_press"] = np.zeros((1, self.num_tick, len(COMPLEX_MOVEMENT)), dtype = np.float32)
        
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
        
        self.state['image'][0, 0:-1] = self.state['image'][0, 1:]
        self.state['image'][0, -1] = np.array(state, dtype=np.float32)/255.0
        self.state['tick'][0, 0:-1] = self.state['tick'][0, 1:]
        self.state['tick'][0, -1] = self.ticks/2000
        self.state["last_press"][0, 0:-1] = self.state["last_press"][0, 1:]
        self.state["last_press"][0, -1, :] = 0.0
        self.state["last_press"][0, -1, action_index] = 1.0
        
        return deepcopy(self.state), all_reward, done, truncated, info
    
    def reset_step(self, action_index):
        
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            self.reset()
        
        return return_state, all_reward, done, truncated, info
        
    def render(self):
        import cv2
        cv2.imshow("render_image", self.state['image'][0, -1])
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
