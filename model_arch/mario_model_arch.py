import torch as th
from torch import nn

class DuelingDqnNet(nn.Module):
    def __init__(self, obs_shape_dict, num_outputs, image_net_block_num = 1, tick_net_block_num = 1, lstm_layer = 3):
        super(DuelingDqnNet, self).__init__()
        extractors = {}
        total_concat_size = 0
        for key, subspace in obs_shape_dict.items():
            if key == "image":
                image_net = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=8, stride=4, padding= 2),
                    nn.LeakyReLU(),
                        # nn.MaxPool2d(kernel_size=3, stride=1),
                        # nn.LayerNorm2d(32)),
                    nn.Conv2d(32, 128, kernel_size=5, stride=3, padding= 1),
                    nn.LeakyReLU(),
                        # nn.MaxPool2d(kernel_size=3, stride=1, padding= 1),
                        # nn.BatchNorm2d(128),
                    nn.Conv2d(128, 256, kernel_size=3, stride=1),
                    nn.LeakyReLU(),
                        # nn.MaxPool2d(kernel_size=3, stride=1, padding= 1),
                        # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                        # nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                        # # nn.BatchNorm2d(256),
                        # # nn.BatchNorm2d(256),
                        # nn.LeakyReLU(),
                    nn.Flatten(),
                        # nn.LeakyReLU(),
                        # nn.LayerNorm(1024),
                        # nn.LeakyReLU(),
                    nn.Linear(9216, 512),
                    nn.LeakyReLU()
                    )
                extractors[key] = image_net
                total_concat_size += 512
            elif key == 'tick':
                tick_net = nn.Sequential(
                    nn.Linear(1, 256),
                    nn.LeakyReLU())
                extractors[key] = tick_net
                total_concat_size += 256
            elif key == 'last_press':
                last_press_net = nn.Sequential(
                    nn.Linear(num_outputs, 512),
                    nn.LeakyReLU())
                extractors[key] = last_press_net
                total_concat_size += 512
                
        self.extractors = nn.ModuleDict(extractors)
        
        # Update the features dim manually
        # self._features_dim = total_concat_size
        self.hidden_size = total_concat_size
        self.lstm_layer = lstm_layer
        self.rnn = nn.LSTM(input_size=total_concat_size, hidden_size=self.hidden_size, num_layers=self.lstm_layer, batch_first=False)

        self.value = nn.Sequential(
            nn.Linear(self.lstm_layer*self.hidden_size*2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1))
        
        self.advantage = nn.Sequential(
            nn.Linear(self.lstm_layer*self.hidden_size*2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_outputs))
                
    def forward(self, observations) -> th.Tensor:
        
        feature_list = []
        for key, obs in observations.items():
            if key == 'image':
                image_feature = [self.extractors[key](obs[:,i].unsqueeze(dim = 1)).unsqueeze(dim = 1) for i in range(obs.shape[1])]
                image_feature = th.cat(image_feature, dim = 1)
                feature_list.append(image_feature)
            else:
                feature_list.append(self.extractors[key](obs))
                
        feature_list_tensor = th.cat(feature_list, dim = -1).transpose(0, 1)
        
        batch_size = feature_list_tensor.shape[1]
        
        __, (hn, cn)= self.rnn(feature_list_tensor)
        
        hn, cn = hn.transpose(0, 1).contiguous().view(batch_size, -1), cn.transpose(0, 1).contiguous().view(batch_size, -1)
        
        
        rnn_output = th.cat([hn,cn], dim = -1)
        
        value = self.value(rnn_output)
        advantage = self.advantage(rnn_output)
        
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
    
        # rnn_output
        
        #     if 'image' in key:
                
        #         # if obs.shape[-1] == 3:
        #         #     obs = obs.permute([0,2,3,1,4])
        #         #     obs = obs.reshape(obs.shape[0], obs.shape[1], obs.shape[2], -1).permute([0,3,1,2])
        #         # t = obs
        #         # for i in range(len(self.extractors['image'])):
        #         #     t = self.extractors['image'][i](t)
        #         #     print(self.extractors['image'][i], t.max(),t.min())
        #         # print("#####")
        #         encoded_image_list.append(self.extractors['image'](obs))
        #     else:
        #         encoded_tensor_list.append(self.extractors[key](obs))
        
        # encoded_image_feature = th.cat(encoded_image_list, dim=1)
        
        # for i in range(len(self.image_net_blocks)):
        #     encoded_image_feature =  encoded_image_feature + self.image_net_blocks[i](encoded_image_feature)
        # # encoded_data_tensor = th.cat(encoded_tensor_list, dim=1)

        # # print(encoded_image_tensor, encoded_data_tensor)

        # # for i in range(len(self.advantage)):
        # #     t = self.advantage[i](t)
        # #     print(self.advantage[i], t.max(),t.min())
        # # print("#####")
        # advantage = self.advantage(encoded_image_feature)
        # # advantage_sum = advantage.clone().detach().sum(dim = 0)
        
        # # self.advantage_memory_tensor += 2*(advantage_sum - advantage_sum.min())/(advantage_sum.max() + 1e-7 - advantage_sum.min()) - 1.0
        
        # # print(self.advantage_memory_tensor)
        # # advantage *= (1.0-self.advantage_memory_tensor*0.1)
        # value = self.value(encoded_image_feature)
        # # return advantage
        # # print("feature",feature, advantage)
        # return value + advantage - advantage.mean(dim=-1, keepdim=True)

    # def reset_noise(self):
    #    self.value[3].reset_noise()
    #    self.value[5].reset_noise()
    #    self.advantage[3].reset_noise()
    #    self.advantage[5].reset_noise()
    
# class Actor(nn.Module):
#     def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(nb_states, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, nb_actions)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
    
    
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         out = self.tanh(out)
#         return out



# class Critic(nn.Module):
#     def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(nb_states, hidden1)
#         self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
#         self.fc3 = nn.Linear(hidden2, 1)
#         self.relu = nn.ReLU()
#         self.init_weights(init_w)
    
#     def init_weights(self, init_w):
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.fc3.weight.data.uniform_(-init_w, init_w)
    
#     def forward(self, xs):
#         x, a = xs
#         out = self.fc1(x)
#         out = self.relu(out)
#         # debug()
#         out = self.fc2(torch.cat([out,a],1))
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out