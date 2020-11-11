import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
############################## DDDQN #################################
######################################################################


class DDDQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DDDQNetwork, self).__init__()

        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc3_val = nn.Linear(hidsize2, 1)

        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc3_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc3_adv(adv)
        return val + adv - adv.mean()
