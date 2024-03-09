import torch as th
from torch import nn
import time
from copy import deepcopy

class ModelWrapper():
    def __init__(
        self, 
        model_arch:nn.Module, 
        optimizer: th.optim, 
        loss_function: nn, 
        obs_shape_dict: dict, 
        act_n: int, 
        learning_rate: float, 
        gamma: float, 
        weight_decay: float, 
        tau: float,
        device: str, 
        model_save_dir: str):
        self.__device = device
        self.input_shape = obs_shape_dict
        self.__act_n = act_n
        
        self.q_net, self.q_net_target = self.__build_model(model_arch, obs_shape_dict, act_n)
        self.optimizer = self.__build_optimizer(optimizer, learning_rate, weight_decay)
        self.loss_func =self.__build_loss_function(loss_function)        
        self.gamma = gamma
        self.tau = tau
        self.model_save_dir = model_save_dir
    
    @property
    def act_n(self):
        return self.__act_n
    @property
    def device(self):
        return self.__device
    
    @device.setter
    def device(self, device: str):
        self.__device = device
    
    def __build_model(self, model_arch:nn.Module, obs_shape_dict: dict, act_n:int):
        q_net = model_arch(obs_shape_dict, act_n)
        q_net_target = model_arch(obs_shape_dict, act_n)
        return q_net.to(self.__device), q_net_target.to(self.__device)
    
    def __build_optimizer(self, optimizer, lr, weight_decay):
        return optimizer(self.q_net.parameters(), lr = lr, weight_decay = weight_decay)
    
    def __build_loss_function(self, loss_function):
        return loss_function().to(device = self.__device)
    
    def numpy2net_tensor(self, states_dict: dict)-> dict:
        states = deepcopy(states_dict)
        for key, value in states_dict.items():
            states[key] = th.FloatTensor(value).to(self.__device)
        return states
    
    def train(self, states, actions, rewards, next_states, dones, truncateds):
        batch_size = actions.shape[0]
        q_next = self.q_net_target(next_states).detach()
        q_targets = rewards + self.gamma * (1 - dones + truncateds) * q_next.max(1)[0].view(batch_size, 1)
        q_values = self.q_net(states).gather(1, actions)
        dqn_loss = self.loss_func(q_values, q_targets)
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        return dqn_loss.item()
     
    def eval(self, states):
        '''
        @return the q_value
        '''
        q_value = self.q_net(states).detach()
        
        return q_value
        
    def target_update(self):
        for target_param, evaluation_param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * evaluation_param.data + (1 - self.tau) * target_param.data)
    def save(self):
        th.save(self.q_net_target.state_dict(),
            self.model_save_dir+"/%s.pt" % (str(time.time_ns())))
    def load(self, file_name:str):
        print("load the model %s" % file_name)
        self.q_net.load_state_dict(th.load(file_name))
        self.q_net_target.load_state_dict(self.q_net.load_state_dict)
    def load_newest(self, model_file_dir: str):
        import os
        load_file_name = '0'
        for file_name in os.listdir(model_file_dir):
            if file_name.endswith(".pt") and file_name > load_file_name:
                load_file_name = file_name

        if load_file_name != '0':
            load_file_path = model_file_dir + '/'+load_file_name
            print("found the newest file %s" % load_file_path, " try to load")
            self.q_net.load_state_dict(th.load(load_file_path))
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            
def configure_optimizers(model: nn.Module, weight_decay:float):
    # Parameters must have a defined order.
    # No sets or dictionary iterations.
    # See https://pytorch.org/docs/stable/optim.html#base-class
    # Parameters for weight decay.
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (th.nn.Linear, th.nn.Conv2d)
    blacklist_weight_modules = (
        th.nn.BatchNorm1d, th.nn.BatchNorm2d, th.nn.LayerNorm, th.nn.Embedding)
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