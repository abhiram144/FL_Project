import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  
from tqdm import tqdm

import torch as th
from torchvision import datasets, transforms


hook = sy.TorchHook(torch) 
client = sy.VirtualWorker(hook, id="client") 
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider") 