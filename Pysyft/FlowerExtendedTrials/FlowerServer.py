from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
# Create an instance of the model and get the parameters
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    
    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "current_round": rnd,  # The current round of federated learning
        "local_epochs": 1 if rnd < 2 else 2,  # 
        "Fit_Config" : 5
    }
    return config

def get_parameters(net) -> List[np.ndarray]:
    print("In server Get Params")
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def evaluate(weights: fl.common.Weights):
    print("In server Evaluate")
    return float(0), {'Evaluate' : 222}

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_eval=1,
    min_fit_clients=1,
    min_eval_clients=1,
    min_available_clients=1,
    initial_parameters= get_parameters(net),
    eval_fn=evaluate,
    on_fit_config_fn=fit_config
)

fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=strategy)