import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import syft as sy

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

hook = sy.TorchHook(torch)
jake = sy.VirtualWorker(hook, id="jake")
john = sy.VirtualWorker(hook, id="john")

train_set = datasets.MNIST(
    "~/.pytorch/MNIST_data/", train=True, download=True, transform=transform)
test_set = datasets.MNIST(
    "~/.pytorch/MNIST_data/", train=False, download=True, transform=transform)

federated_train_loader = sy.FederatedDataLoader(
    train_set.federate((jake, john)), batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(0, 5):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        # send the model to the client device where the data is present
        model.send(data.location)
        # training the model
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # get back the improved model
        model.get()
        if batch_idx % 100 == 0:
            # get back the loss
            loss = loss.get()
            print('Epoch: {:2d} [{:5d}/{:5d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1,
                batch_idx * 64,
                len(federated_train_loader) * 64,
                100. * batch_idx / len(federated_train_loader),
                loss.item()))

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(
            output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.argmax(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss,
    correct,
    len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))