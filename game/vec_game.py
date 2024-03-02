from multiprocessing import Process, Pipe
import os
import asyncio
import numpy as np

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
            state, info = one_game.reset()
            conn2.send((state, info))
        elif msg == 'render':
            one_game.render()

class vec_game():
    def __init__(self, nums:int, game: object, monitor_file_dir:str, obs_shape_dict:dict):
        self.game_class = game
        self.nums = nums
        self.obs_shape_dict = obs_shape_dict
        self.conn1_list = []
        for i in range(nums):
            conn1, conn2 = Pipe(True)
            monitor_file_path = os.path.join(monitor_file_dir,str(i)+".csv")
            sub_process = Process(target=game_worker, args=(self.game_class, monitor_file_path, conn2))
            sub_process.start()
            self.conn1_list.append(conn1)
    def reset(self):
        states = {}
        for key in self.obs_shape_dict.keys():
            states[key] = []
        infos = []
        
        for i in range(self.nums):
            self.conn1_list[i].send(('reset', 0))
        for i in range(self.nums):
            state, info = self.conn1_list[i].recv()
            for key, value in state.items():
                states[key].append(value)
            infos.append(info)
            
        for key, value in states.items():
            states[key] = np.concatenate(value, axis=0, dtype = np.float32)
            
        return states, infos
    def step(self, action_vec):
        states = {}
        for key in self.obs_shape_dict.keys():
            states[key] = []
        rewards = []
        dones = []
        truncateds = [] 
        infos = []
        for i in range(self.nums):
            self.conn1_list[i].send(('reset_step', action_vec[i]))
            
        for i in range(self.nums):
            # TODO: 允许异步执行
            state, reward, done, truncated, info = self.conn1_list[i].recv()
            for key, value in state.items():
                states[key].append(value)
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