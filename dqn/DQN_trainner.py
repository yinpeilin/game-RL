import numpy as np
import torch
import torch.nn as nn
import time
# 神经网络模型训练
def configure_optimizers(model: nn.Module, weight_decay:float):
    # Parameters must have a defined order.
    # No sets or dictionary iterations.
    # See https://pytorch.org/docs/stable/optim.html#base-class
    # Parameters for weight decay.
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    blacklist_weight_modules = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if pn.endswith('bias'):
                    # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert len(
        inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )
    optim_groups = [
        {"params": [param_dict[pn]
                    for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn]
                    for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups


class DQNTrainer():
    # 初始化传入环境的初始化信息与状态，
    # agent可操作的动作数量，
    # γ衰减率
    # 学习率
    # 利用探索率
    # model->target的步数
    def __init__(
            self,
            env_num,
            obs_shape_dict,
            act_n,
            model_arch,
            buffer_arch,
            gamma=0.9,
            tau=1.0,
            lr=1e-3,
            weight_decay=1e-7,
            eps_clip=0.5,
            buffer_size=50000,
            target_update=1000,
            device='cuda',
            model_save_dir="./model"
    ):
        # 创建记忆库
        self.replay_buffer = buffer_arch(buffer_size=buffer_size, obs_shape_dict= obs_shape_dict)
        self.obs_shape_dict = obs_shape_dict
        self.act_n = act_n
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.eps_clip = eps_clip
        self.target_update = target_update

        # 创建模型
        self.device = device
        self.q_net, self.q_net_target = self.__build_model(model_arch)
        # 使用Adam优化器
        # optimizer_group = configure_optimizers(self.q_net)
        # self.optimizer = torch.optim.Adam(optimizer_group,
        #                                    lr=self.lr, weight_decay=WEIGHT_DECAY)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                        lr=self.lr, weight_decay=weight_decay)
        # 所跑轮数
        self.train_count = 0
        self.states_len = env_num
        self.index_list = [i for i in range(self.states_len)]
        self.criterion = torch.nn.MSELoss().to(device=self.device)
        self.model_save_dir = model_save_dir
    def __build_model(self, model_arch):
        q_net = model_arch(self.obs_shape_dict, self.act_n)
        q_net_target = model_arch(self.obs_shape_dict, self.act_n)
        q_net_target.load_state_dict(q_net.state_dict())
        
        q_net.train()
        q_net_target.eval()
        return q_net.to(self.device), q_net_target.to(self.device)

    def choose_action(self, states):
        # 该状态动作选择
        temp = torch.rand(1)
        if temp < self.eps_clip:
            actions = np.random.randint(0, self.act_n, (self.states_len,))
        else:
            states_dict = {}
            for key in self.obs_shape_dict.keys():
                states_dict[key] = torch.cat([torch.tensor(states[i][key]).unsqueeze(dim = 0) for i in self.index_list], dim = 0).to(self.device)
            q_value = self.q_net(states_dict)
            actions = q_value.argmax(dim=1).cpu().numpy()
        return actions
    # 存储，根据memory类,存储当前状态，价值，动作后状态，是否结束
    def store(self, states, actions, rewards, next_states, dones, truncateds):
        states_save = {}
        next_states_save = {}
        for key in self.obs_shape_dict.keys():
            states_save[key] = torch.cat([torch.tensor(states[i][key]).unsqueeze(dim = 0) for i in self.index_list], dim = 0)
            next_states_save[key] = torch.cat([torch.tensor(next_states[i][key]).unsqueeze(dim = 0) for i in self.index_list], dim = 0)
        actions = torch.tensor(actions).type(torch.int32)
        rewards = torch.tensor(rewards).type(torch.FloatTensor)
        dones = torch.tensor(dones).type(torch.FloatTensor)
        truncateds = torch.tensor(truncateds).type(torch.FloatTensor)

        self.replay_buffer.add(states_save, actions, rewards, next_states_save, dones, truncateds)

    def learn(self, batch_size=128):
        states, actions, rewards, next_states, dones, truncateds = self.replay_buffer.sample(batch_size)
        for key in self.obs_shape_dict:
            states[key] = states[key].to(self.device)
            next_states[key] = next_states[key].to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        truncateds = truncateds.to(self.device)
        # next_actions = self.q_net(next_states).argmax(dim = 1)
        # q_targets = rewards + self.gamma  * (1.0 - dones + truncateds)*self.q_net_target(next_states)[torch.arange(0, batch_size), next_actions]  # Q值
        q_targets = rewards + self.gamma * (1.0- dones + truncateds)*(self.q_net_target(next_states).detach()).max(1)[0]
        q_values = self.q_net(states)[torch.arange(0, batch_size), actions]
        dqn_loss = self.criterion(q_values, q_targets)
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        if self.train_count % self.target_update == 0:
            # print("target model update!")
            # target_net_state_dict = self.q_net_target.state_dict()
            # policy_net_state_dict = self.q_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key] * \
            #         self.tau + target_net_state_dict[key]*(1-self.tau)
            # self.q_net_target.load_state_dict(target_net_state_dict)
            self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.train_count += 1
        return dqn_loss.item()
    def save(self):
        # 保存模型
        torch.save(self.q_net_target.state_dict(),
            self.model_save_dir+"/%s.pt" % (str(time.time_ns())))

    def load(self, file_name):
        print("load the model %s" % file_name)
        self.q_net.load_state_dict(torch.load(file_name))
        self.q_net_target.load_state_dict(torch.load(file_name))

    def load_newest(self, file_dir):
        import os
        load_file_name = '0'
        for file_name in os.listdir(file_dir):
            if file_name.endswith(".pt") and file_name > load_file_name:
                load_file_name = file_name

        if load_file_name != '0':
            load_file_path = file_dir + '/'+load_file_name
            print("found the newest file %s" % load_file_path, " try to load")
            self.q_net.load_state_dict(torch.load(load_file_path))
            self.q_net_target.load_state_dict(self.q_net.state_dict())
