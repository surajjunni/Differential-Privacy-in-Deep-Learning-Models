import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from opacus.layers import DPLSTM
from opacus import PrivacyEngine

# Define constants
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 20
LEARNING_RATE = 0.01
MOMENTUM = 0.5
MAX_GRAD_NORM = 1.2

# Define transformations for the dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Download and load the training and test data
train_dataset = datasets.MNIST(
    "../data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    "../data", train=False, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=75, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = DPLSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# Initialize the model, loss function, and optimizer
model = LSTMModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=MAX_GRAD_NORM,
)

# Train the model
for epoch in range(1, EPOCHS + 1):
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
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        predicted = (outputs >= 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {correct / total * 100:.2f}%")

