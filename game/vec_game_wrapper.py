from multiprocessing import Process, Pipe
import os
import asyncio
import numpy as np


def check_env(game_class, obs_shape_dict):
    
    game = game_class()
    for key in obs_shape_dict.keys():
        if game.state[key].shape != obs_shape_dict[key]:
            raise ValueError("the {} setting may not be same as the game setting, \
                            the game shape is {}, the setting shape is {}".format(key, game.state[key], obs_shape_dict[key]))
    del game
    
def game_worker(game, monitor_file_path, conn2):
    one_game = game(monitor_file_path = monitor_file_path)
    while True:
        msg, arg = conn2.recv()
        if msg == 'step':
            state, reward, done, truncated, info = one_game.step(arg)
            conn2.send((state, reward, done, truncated, info))
        if msg == 'reset_step':
            state, reward, done, truncated, info = one_game.reset_step(arg)
            conn2.send((state, reward, done, truncated, info))
        elif msg == 'reset':
            state, info = one_game.reset(arg)
            conn2.send((state, info))
        elif msg == 'render':
            one_game.render()

class VecGameWrapper():
    # TODO: get the async method finished
    def __init__(self, nums:int, game_class: object, monitor_file_dir:str, obs_shape_dict:dict):
        check_env(game_class, obs_shape_dict)
        
        self.game_class = game_class
        self.nums = nums
        self.obs_shape_dict = obs_shape_dict
        self.conn1_list = []
        
        if not os.path.exists(monitor_file_dir):
            os.mkdir(monitor_file_dir)
        
        for i in range(nums):
            conn1, conn2 = Pipe(True)
            monitor_file_path = os.path.join(monitor_file_dir,str(i)+".csv")
            sub_process = Process(target=game_worker, args=(self.game_class, monitor_file_path, conn2))
            sub_process.start()
            self.conn1_list.append(conn1)
            
    @property
    def num(self)-> int:
        return self.nums
    
    def reset(self):
        states = {}
        infos = []
        
        for key in self.obs_shape_dict.keys():
            states[key] = []
        for i in range(self.nums):
            self.conn1_list[i].send(('reset', 0))
        for i in range(self.nums):
            state, info = self.conn1_list[i].recv()
            for key, value in state.items():
                states[key].append(np.expand_dims(value, axis=0))
            infos.append(info)
            
        for key, value in states.items():
            states[key] = np.concatenate(value, axis=0, dtype = np.float32)
            
        return states, infos
    def step(self, action_vec):
        
        assert len(action_vec) == self.nums, "the action_vec shound be the same as the game numbers"
        
        states = {}
        rewards = []
        dones = []
        truncateds = [] 
        infos = []
        for key in self.obs_shape_dict.keys():
            states[key] = []
        for i in range(self.nums):
            self.conn1_list[i].send(('reset_step', action_vec[i]))
            
        for i in range(self.nums):
            state, reward, done, truncated, info = self.conn1_list[i].recv()
            for key, value in state.items():
                states[key].append(np.expand_dims(value, axis=0))
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
        for key, value in states.items():
            states[key] = np.concatenate(value, axis=0, dtype = np.float32)
            
            
        return states, rewards, dones, truncateds, infos
        # return asyncio.run(self.async_step(action_vec))
    def render(self):
        for i in range(self.nums):
            self.conn1_list[i].send(('render', 0))
            
        # return asyncio.run(self.async_render())
    async def send_msg(self, index ,msg, arg):
        self.conn1_list[index].send((msg, arg))
        
    async def recv_reset_msg(self, index, states, infos):
        state, info = self.conn1_list[index].recv()
        states.append(state)
        infos.append(info)
        
    async def recv_step_msg(self, index, states, rewards, dones, truncateds, infos):
        state, reward, done, truncated, info = self.conn1_list[index].recv()
        states.append(state)
        rewards.append(reward)
        dones.append(done)
        truncateds.append(truncated)
        infos.append(info)
    async def async_step(self,  action_vec):
        states = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        send_tasks = [asyncio.create_task(self.send_msg(index, 'reset_step', action_vec[index])) for index in range(self.nums)]
        await asyncio.wait(send_tasks)
        recv_tasks = [asyncio.create_task(self.recv_step_msg(index, states, rewards, dones, truncateds, infos)) for index in range(self.nums)]
        await asyncio.wait(recv_tasks)
        return states, rewards, dones, truncateds, infos
    async def async_reset(self):
        states = []
        infos = []
        send_tasks = [asyncio.create_task(self.send_msg(index, 'reset', 0)) for index in range(self.nums)]
        await asyncio.wait(send_tasks)
        recv_tasks = [asyncio.create_task(self.recv_reset_msg(index, states, infos)) for index in range(self.nums)]
        await asyncio.wait(recv_tasks)
        return states, infos
    async def async_render(self):
        send_tasks = [asyncio.create_task(self.send_msg(index, 'render', 0)) for index in range(self.nums)]
        await asyncio.wait(send_tasks)