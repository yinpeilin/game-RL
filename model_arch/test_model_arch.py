import torch as th
from torch import nn

class TestModelArch(nn.Module):
    def __init__(self, obs_shape_dict, num_outputs, box_observation_block_num = 1):
        super(TestModelArch, self).__init__()
        extractors = {}
        total_concat_size = 0
        for key, subspace in obs_shape_dict.items():
            if key == "box_observation":
                box_observation_net = nn.ModuleList()
                box_observation_net.append(
                    nn.Sequential(
                        nn.Linear(4, 512),
                        nn.LeakyReLU()
                    )
                )
                
                for i in range(box_observation_block_num):
                    image_block_net = nn.Sequential(
                        nn.Linear(512, 512),
                        nn.LeakyReLU(),
                        nn.Linear(512, 512),
                        nn.LeakyReLU()
                    )
                    box_observation_net.append(image_block_net)
                extractors[key] = box_observation_net
                total_concat_size += 512
                

        self.extractors = nn.ModuleDict(extractors)        # Update the features dim manually
        self.value = nn.Sequential(
            nn.Linear(total_concat_size*list(obs_shape_dict.values())[0][0], 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1))
        
        self.advantage = nn.Sequential(
            nn.Linear(total_concat_size*list(obs_shape_dict.values())[0][0], 256),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_outputs))
        
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        seq_len = observations["box_observation"].shape[1]
        batch_size = observations["box_observation"].shape[0]
        for i in range(seq_len):
            feature_list = [ ]
            for key, obs in observations.items():
                if key == "box_observation":
                    feature = self.extractors[key][0](obs[:,i].unsqueeze(dim = 1))
                    for j in range(1, len(self.extractors[key])):
                        feature = feature + self.extractors[key][j](feature)
                    feature = feature.unsqueeze(dim = 1)
                feature_list.append(feature)
            feature_list = th.cat(feature_list, dim = -1)
            encoded_tensor_list.append(feature_list)
        encoded_tensor_list = th.cat(encoded_tensor_list, dim = 1).squeeze(dim = -2).view(batch_size, -1)
        value = self.value(encoded_tensor_list)
        advantage = self.advantage(encoded_tensor_list)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)