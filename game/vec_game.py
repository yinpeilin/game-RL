from multiprocessing import Process, Pipe
import asyncio

def game_worker(game, index, conn2):
    one_game = game(monitor_file_path = 'dqn/result/monitor/'+str(index)+'.csv')
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
    def __init__(self, nums:int, game: object):
        self.game_class = game
        self.nums = nums
        self.conn1_list = []
        for i in range(nums):
            conn1, conn2 = Pipe(True)
            sub_process = Process(target=game_worker, args=(self.game_class, i, conn2))
            sub_process.start()
            self.conn1_list.append(conn1)
            
    def reset(self):
        states = []
        infos = []
        for i in range(self.nums):
            self.conn1_list[i].send(('reset', 0))
        
        for i in range(self.nums):
            state, info = self.conn1_list[i].recv()
            states.append(state)
            infos.append(info)
        
        return states, infos
    
    def step(self, action_vec):
        states = []
        rewards = []
        dones = []
        truncateds = [] 
        infos = []
        for i in range(self.nums):
            self.conn1_list[i].send(('reset_step', action_vec[i]))
            
        for i in range(self.nums):
            # 允许异步执行
            state, reward, done, truncated, info = self.conn1_list[i].recv()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
            
        return states, rewards, dones, truncateds, infos
    
    def render(self):
        for i in range(self.nums):
            self.conn1_list[i].send(('render', 0))
    
    
