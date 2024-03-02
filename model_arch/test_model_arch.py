import torch as th
from torch import nn
class TestModelArch(nn.Module):
    def __init__(self, obs_shape_dict, num_outputs):
        super(TestModelArch, self).__init__()
        extractors = {}
        total_concat_size = 0
        for key, subspace in obs_shape_dict.items():
            if key == "box_observation":
                extractors[key] = nn.Sequential(
                        nn.Linear(4, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                    )
                total_concat_size += 1024
        self.extractors = nn.ModuleDict(extractors)        # Update the features dim manually
        self.value = nn.Sequential(
            nn.Linear(total_concat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1))
        self.advantage = nn.Sequential(
            nn.Linear(total_concat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
            )
    def forward(self, observations) -> th.Tensor:
        feature_list = [self.extractors[key](obs) for key, obs in observations.items()]
        feature_list_tensor = th.cat(feature_list, dim = -1).squeeze(dim = -2)
        value = self.value(feature_list_tensor)
        advantage = self.advantage(feature_list_tensor)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)