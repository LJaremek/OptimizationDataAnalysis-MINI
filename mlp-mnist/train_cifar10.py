import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from bitorch.layers import convert
from bitorch import RuntimeMode
import bitorch.layers as qnn
import bitorch_engine

from datasets import CIFAR10, BasicDataset

bitorch_engine.initialize()


class BinarizedConvNet(nn.Module):
    def __init__(self):
        super(BinarizedConvNet, self).__init__()
        self.conv1 = qnn.QConv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = qnn.QConv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = qnn.QConv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = qnn.QConv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = qnn.QConv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = qnn.QConv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = qnn.QLinear(512*4*4, 1024)
        self.fc2 = qnn.QLinear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)
        # self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        # x = self.drop(x)
        x = self.fc3(x)
        return x


def train(
        model: nn.Module, train_loader: BasicDataset, optimizer, criterion,
        device: str = "cpu", epoch: int = 1
        ) -> None:

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/"
                f"{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
                )


def test(
        model: nn.Module, test_loader: BasicDataset, criterion,
        device: str = "cpu"
        ) -> None:

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/"
        f"{len(test_loader.dataset)} "
        f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
        )


def main():
    # parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    lr = 0.000001

    # model, criterion, optimizer
    model = BinarizedConvNet().to(device)
    model = convert(model, RuntimeMode.DEFAULT, device, False)
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

    print(train_loader.dataset.get_transform())

    # train process
    for epoch in range(1, 11):
        train(model, train_loader, optimizer, criterion, device, epoch)
        torch.save(model.state_dict(), f"cifar10_{epoch}.pt")
        test(model, test_loader, criterion, device)

    test(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
