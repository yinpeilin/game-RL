import gymnasium as gym
from gymnasium import spaces
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_EASY
import numpy as np
import cv2
import csv
from copy import deepcopy

global_world_index_list = [1, 2, 3, 4, 5, 6, 7, 8]
global_stage_index_list = [1, 2, 3, 4]
version_index_list = [0, 1, 2, 3]
class mario_env(gym.Env):
    def __init__(self, world_index_list = None, stage_index_list = None, version_index = None, frame_tick = 1, num_tick = 4, monitor_file = 'test.csv'):
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
        # game = gym.make('SuperMarioBrosRandomStages-v{}'.format(version_index),
                # stages=str_temp_list)
       
        game = gym.make('SuperMarioBros-v0')
        self.game = JoypadSpace(game, RIGHT_EASY)

        self.num_tick = num_tick
        self.frame_tick = frame_tick
        
        self.state = {
            'image': np.zeros((self.num_tick, 100,100), dtype=np.float32),
            "tick": np.zeros((self.num_tick, 1), dtype = np.float32),
            "last_press": np.zeros(( self.num_tick, len(RIGHT_EASY)), dtype = np.int32)   
        }
        
        self.action_space = spaces.Discrete(len(RIGHT_EASY))
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(self.num_tick, 100, 100), dtype=np.float32),
            "tick": spaces.Box(low=0, high=1, shape=(self.num_tick, 1), dtype=np.float32),
            "last_press":spaces.Box(low=0, high=1, shape=(self.num_tick,len(RIGHT_EASY)), dtype=np.float32)
        })
        self.monitor_file_path = monitor_file
        self.ticks = 0.0
        self.all_reward = 0.0
    def reset(self, seed = None):
        state, info = self.game.reset()
        state = cv2.resize(state, (100, 100))
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        
        self.state['image'] = np.zeros((self.num_tick, 100, 100), dtype=np.float32)
        self.state['image'][-1] = np.array(state, dtype=np.float32)/255.0
        
        
        monitor_file = csv.writer(open(self.monitor_file_path, 'a', newline=''))
        monitor_file.writerow([int(self.ticks), int(self.all_reward)])
        self.ticks = 0.0
        
        self.state['tick'] = np.zeros((self.num_tick, 1), dtype = np.float32)
        self.state["last_press"] = np.zeros(( self.num_tick, 3), dtype = np.float32)
        
        
        self.all_reward = 0.0
        
        return self.state, {"test":123}
    
    def step(self, action_index):
        
        all_reward = 0.0
        # self.state[self.frame_tick:] = self.state[0:-self.frame_tick]
        
        for i in range(self.frame_tick):
            state, reward, done, truncated, info = self.game.step(action_index)
            state = cv2.resize(state, (100, 100))
            # state.swapaxes(0，2)
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            # self.state[self.frame_tick - 1 - i] = state
            all_reward += reward
            self.ticks += 1.0
            if self.ticks >= 2000:
                truncated = True
                done = True
            if done == True:
                # t = self.frame_tick - 2 - i
                # while t>=0:
                #     self.state[t] = state
                #     t-=1
                break
        self.all_reward += all_reward
        
        self.state['image'][0:-1] = self.state['image'][1:]
        self.state['image'][-1] = np.array(state, dtype=np.float32)/255.0
        self.state['tick'][0:-1] = self.state['tick'][1:]
        self.state['tick'][-1] = self.ticks/2000
        self.state["last_press"][0:-1] = self.state["last_press"][1:]
        self.state["last_press"][-1][:] = 0.0
        self.state["last_press"][-1][action_index] = 1.0
        
        return self.state, all_reward, done, truncated, info
    
    def reset_step(self, action_index):
        
        return_state, all_reward, done, truncated, info = self.step(action_index)
        if done == True:
            return_state = deepcopy(self.state)
            self.reset()
        return return_state, all_reward, done, truncated, info
        
    def render(self):
        import cv2
        cv2.imshow("render_image", self.state['image'][-1])
        cv2.waitKey(10)

def make_mario_env():
    
    return mario_env()

if __name__ == '__main__':
    env = mario_env(frame_tick=8)
    done = True
    while True:
        if done == True:
            env.reset()
        state, reward, done, truncated, info = env.step(env.action_space.sample())
        env.render()
