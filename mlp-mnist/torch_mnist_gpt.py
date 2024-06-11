import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Define the quantized MLP model
class QuantizedMLP(nn.Module):
    def __init__(self):
        super(QuantizedMLP, self).__init__()
        self.quant = torch.quantization.QuantStub()  # Quantization stub
        self.dequant = torch.quantization.DeQuantStub()  # De-quantization stub
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(32, 10),
        )
        
    def forward(self, x):
        x = self.quant(x)  # Quantize the input
        x = self.layers(x)
        x = self.dequant(x)  # De-quantize the output
        output = F.log_softmax(x, dim=1)
        return output

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define training and evaluation functions
def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return running_loss / len(train_loader), correct / total

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return running_loss / len(test_loader), correct / total

# Create the quantized MLP model
model = QuantizedMLP()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model, inplace=False)
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_quantized.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss, train_accuracy = train(model_quantized, train_loader, optimizer, criterion)
    test_loss, test_accuracy = evaluate(model_quantized, test_loader, criterion)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
torch.save(model_quantized.state_dict(), 'quantized_mlp_mnist.pth')
