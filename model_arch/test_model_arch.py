import torch as th
from torch import nn
class TestModelArch(nn.Module):
    def __init__(self, obs_shape_dict, num_outputs):
        super(TestModelArch, self).__init__()
        extractors = {}
        total_concat_size = 0
        for key, subspace in obs_shape_dict.items():
            if key == "box_observation":
                box_observation_net = nn.ModuleList()
                box_observation_net.append(
                    nn.Sequential(
                        nn.Linear(4, 50),
                        nn.ReLU(),
                    )
                )
                extractors[key] = box_observation_net
                total_concat_size += 50
        self.extractors = nn.ModuleDict(extractors)        # Update the features dim manually
        # self.value = nn.Sequential(
        #     nn.Linear(total_concat_size*list(obs_shape_dict.values())[0][0], 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 1))
        self.advantage = nn.Sequential(
            nn.Linear(total_concat_size*list(obs_shape_dict.values())[0][0], 50),
            nn.ReLU(),
            nn.Linear(50, num_outputs))
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        seq_len = observations["box_observation"].shape[1]
        for i in range(seq_len):
            feature_list = []
            for key, obs in observations.items():
                if key == "box_observation":
                    feature = self.extractors[key][0](obs[:,i]).unsqueeze(dim = 1)
                feature_list.append(feature)
            feature_list = th.cat(feature_list, dim = -1)
            encoded_tensor_list.append(feature_list)
        encoded_tensor = th.cat(encoded_tensor_list, dim = 1).squeeze(dim = -2)
        # value = self.value(encoded_tensor)
        advantage = self.advantage(encoded_tensor)
        
        return advantage
        # return value + advantage - advantage.mean(dim=-1, keepdim=True)