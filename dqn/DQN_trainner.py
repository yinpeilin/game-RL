import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from game.vec_game_wrapper import VecGameWrapper
from model.model_wrapper import DuelingDqnModelWrapper
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
# 神经网络模型训练

class DQNTrainer():
    # hava a
    # 有vec game ———— 产生数据
    # -->step update | all log update
    # -->储存数据
    # 有replay_buffer
    # --> batch_size
    # 有dqn ———— 用数据训练
    # --> dqn->vec_game_action:choose_action
    
    
    # 初始化传入环境的初始化信息与状态，
    # agent可操作的动作数量，
    # γ衰减率
    # 学习率
    # 利用探索率
    # model->target的步数
    def __init__(
            self,
            start_eps_rate: float,
            end_eps_rate: float,
            eps_decrease_ratio: float,
            eps_clip_step: int,
            target_update_step: int,
            train_model_step: int,
            tqdm_step: int,
            tensorboard_step:int,
            save_model_step:int,
            batch_size: int,
            tensorboard_dir: str
    ):
        # the setting init
        
        self.start_eps_rate = start_eps_rate
        self.end_eps_ratio = end_eps_rate
        self.eps_decrease_ratio = eps_decrease_ratio
        
        self.eps = self.start_eps_rate
        
        self.target_update_step = target_update_step
        self.train_model_step = train_model_step
        self.tqdm_step = tqdm_step
        self.tensorboard_step = tensorboard_step
        self.save_model_step = save_model_step
        self.eps_clip_step = eps_clip_step
        
        self.batch_size = batch_size
        self.tensorboard_writer = SummaryWriter(tensorboard_dir)
        
        # 所跑轮数
        self.step = 0
    
        self.vec_game = None
        self.model_wrapper = None
        self.replay_buffer = None
    def vec_game_init(self,  nums, game_class, monitor_file_dir, obs_shape_dict):
        self.vec_game = VecGameWrapper(nums, game_class, monitor_file_dir, obs_shape_dict)
        
    def model_wrapper_init(self, model_arch, optimizer, loss_function, obs_shape_dict, 
        act_n, learning_rate, gamma, weight_decay, tau, device, model_save_dir):
        self.model_wrapper = DuelingDqnModelWrapper(model_arch, optimizer, loss_function, obs_shape_dict, 
                                                    act_n, learning_rate, gamma, weight_decay, tau, device, model_save_dir)
        self.model_wrapper.load_newest(model_save_dir)
    def replay_buffer_init(self, buffer_arch, buffer_size, obs_shape_dict):
        self.replay_buffer = buffer_arch(buffer_size=buffer_size, obs_shape_dict= obs_shape_dict)
        
    def update_all(self, total_step: int, need_train: bool, need_render = bool):
        if self.vec_game == None or self.model_wrapper == None or self.replay_buffer == None:
            raise AttributeError('need to init first')
        
        with tqdm(total = total_step) as t:
            states, __ = self.vec_game.reset()
            
            tqdm_loss = 0.0
            tqdm_reward = 0.0
            tensorboard_loss = 0.0
            tensorboard_reward = 0.0
            
            for i in range(1, total_step):
                actions = self.__choose_action(states=states)
                next_states, rewards, dones, truncateds, __ = self.vec_game.step(actions)
                if need_render:
                    self.vec_game.render()
                
                self.__store(states, actions, rewards, next_states, dones, truncateds)
                states = next_states
                
                rewards_sum = sum(rewards)
                tqdm_reward += rewards_sum
                tensorboard_reward += rewards_sum
                if need_train:
                    if i % self.train_model_step == 0:
                        loss = self.__learn(self.batch_size)
                        tqdm_loss += loss
                        tensorboard_loss += loss
                    if i % self.target_update_step == 0:
                        self.model_wrapper.target_update()
                    if i % self.tqdm_step == 0:
                        t.update(self.tqdm_step)
                        t.set_postfix(loss=tqdm_loss / self.tqdm_step,
                                    reward=tqdm_reward / self.tqdm_step, eps=self.eps)
                        tqdm_loss = 0.0
                        tqdm_reward = 0.0
                    if i % self.tensorboard_step == 0:
                        self.tensorboard_writer.add_scalar(
                                "loss_mean", tensorboard_loss / self.tensorboard_step, i)
                        self.tensorboard_writer.add_scalar(
                                "reward_mean", tensorboard_reward / self.tensorboard_step, i)
                        self.tensorboard_writer.add_scalar("eps_clip", self.eps, i)
                        tensorboard_loss = 0.0
                        tensorboard_reward = 0.0
                    if i % self.save_model_step == 0:
                        self.model_wrapper.save()
                    if i % self.eps_clip_step == 0:
                        self.eps *= self.eps_decrease_ratio
                        self.eps = self.end_eps_ratio if self.eps < self.end_eps_ratio else self.eps
    
    def __choose_action(self, states:dict):
        # 该状态动作选择
        if np.random.uniform() < self.eps:
            actions = np.random.randint(0, self.model_wrapper.act_n, self.vec_game.num)
        else:
            states_dict = self.model_wrapper.numpy2net_tensor(states)
            q_value = self.model_wrapper.eval(states_dict)
            actions = q_value.argmax(dim=1).cpu().numpy()
        return actions
    
    def __store(self, states, actions, rewards, next_states, dones, truncateds):
        actions = np.array(actions, dtype= np.int64)
        rewards = np.array(rewards, dtype= np.float32)
        dones = np.array(dones, dtype= np.float32)
        truncateds = np.array(truncateds, dtype= np.float32)
        self.replay_buffer.add(states, actions, rewards, next_states, dones, truncateds)

    def __learn(self, batch_size=128):
        if batch_size > self.replay_buffer.size:
            warnings.warn("the batch_size > the replay_buffer data collected, so the learning will not work", RuntimeWarning)
            return 0.0
        
        states, actions, rewards, next_states, dones, truncateds = self.replay_buffer.sample(batch_size)
        
        for key in states.keys():
            states[key] = states[key].to(self.model_wrapper.device)
            next_states[key] = next_states[key].to(self.model_wrapper.device)
        actions = actions.unsqueeze(dim = -1).to(self.model_wrapper.device)
        rewards = rewards.unsqueeze(dim = -1).to(self.model_wrapper.device)
        dones = dones.unsqueeze(dim = -1).to(self.model_wrapper.device)
        truncateds = truncateds.unsqueeze(dim = -1).to(self.model_wrapper.device)
        # weights = weights.unsqueeze(dim = -1).to(self.device)
        
        # next_actions = self.q_net(next_states).argmax(dim = 1).unsqueeze(dim = -1)
        # q_targets = rewards + self.gamma  * (1.0 - dones + truncateds)*self.q_net_target(next_states).gather(1, next_actions)
        # q_targets = q_targets.detach()
        
        dqn_loss = self.model_wrapper.train(states, actions, rewards, next_states, dones, truncateds)
        # self.replay_buffer.update_priorities(indices, update_priors.cpu().numpy())
        
        return dqn_loss
    
