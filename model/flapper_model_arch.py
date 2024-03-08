import torch as th
from torch import nn

class DuelingDqnNet(nn.Module):
    def __init__(self, obs_shape_dict, num_outputs):
        super(DuelingDqnNet, self).__init__()
        extractors = {}
        total_concat_size = 0
        for key, subspace in obs_shape_dict.items():
            if key == "image":
                self.seq_len = subspace[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=5, stride=3),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=5, stride=3),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(6400, 512),
                    nn.ReLU(),
                    )
                total_concat_size += 512
            elif key == 'tick':
                tick_net = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(),
                    )
                extractors[key] = tick_net
                total_concat_size += 16
            elif key == 'last_press':
                last_press_net = nn.Sequential(
                    nn.Linear(num_outputs, 16),
                    nn.ReLU()
                    )
                extractors[key] = last_press_net
                total_concat_size += 16
                
        self.extractors = nn.ModuleDict(extractors)
        
        # Update the features dim manually
        # self._features_dim = total_concat_size
        # self.hidden_size = total_concat_size
        # self.lstm_layer = lstm_layer
        # self.rnn = nn.LSTM(input_size=total_concat_size, hidden_size=self.hidden_size, num_layers=self.lstm_layer, batch_first=False)
        self.value = nn.Sequential(
            nn.Linear(self.seq_len * total_concat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )
        self.advantage = nn.Sequential(
            nn.Linear(self.seq_len * total_concat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
            )
    def forward(self, observations) -> th.Tensor:
        feature_list = []
        for key, obs in observations.items():
            if key == 'image':
                image_feature = [self.extractors[key](obs[:,i].unsqueeze(dim = 1)).unsqueeze(dim = 1) for i in range(obs.shape[1])]
                image_feature = th.cat(image_feature, dim = 1)
                feature_list.append(image_feature)
            else:
                feature_list.append(self.extractors[key](obs))
        feature_list_tensor = th.cat(feature_list, dim = -1).squeeze(dim = 1)
        
        feature_list_tensor = feature_list_tensor.view(feature_list_tensor.shape[0], -1)
        
        value = self.value(feature_list_tensor)
        advantage = self.advantage(feature_list_tensor)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)