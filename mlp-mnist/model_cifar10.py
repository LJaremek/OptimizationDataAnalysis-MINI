import torch.optim as optim
import torch.nn as nn
import torch

import bitorch.layers as qnn
import bitorch_engine

from datasets import CIFAR10

bitorch_engine.initialize()


class ConvNetClassic(nn.Module):
    def __init__(self):
        super(ConvNetClassic, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2, 0)

        self.conv2 = nn.Conv2d(256, 320, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(320)

        self.conv3 = nn.Conv2d(320, 384, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 256, 5, 1, 2)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 10)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        x = x.view(-1, 256 * 8 * 8)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class ConvNetBinary(nn.Module):
    def __init__(self):
        super(ConvNetBinary, self).__init__()
        self.conv1 = qnn.QConv2d(3, 256, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2, 0)

        self.conv2 = qnn.QConv2d(256, 320, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(320)

        self.conv3 = qnn.QConv2d(320, 384, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = qnn.QConv2d(384, 256, 5, 1, 2)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = qnn.QLinear(256 * 8 * 8, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 10)

        self.activation = qnn.QActivation()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)

        x = x.view(-1, 256 * 8 * 8)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    lr = 0.0001

    # model, criterion, optimizer
    model = ConvNetBinary().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=lr)

    # train/test kwargs
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}

    if device == "cuda":
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True
            }

        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # dataset
    train_dataset, test_dataset = CIFAR10.get_train_and_test(
        "./cifar10",
        download=True
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    for epoch in range(1, 11):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(
            f"Train Epoch: {epoch} [{batch_idx * len(data)}/"
            f"{len(train_loader.dataset)} "
            f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
            f"Loss: {loss.item():.6f}"
        )

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True).view(-1)
                correct += (pred == target).sum().item()
                # correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/"
            f"{len(test_loader.dataset)} "
            f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
        )
