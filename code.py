import warnings
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision.datasets import CIFAR10
from torchvision import models
from opacus.validators import ModuleValidator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


warnings.simplefilter("ignore")

MAX_GRAD_NORM = 1.0
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 50

LR = 1e-3
BATCH_SIZE = 8
#MAX_PHYSICAL_BATCH_SIZE = 16

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = Compose([
    ToTensor(),
    Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

DATA_ROOT = '../cifar10'

train_dataset = CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(num_classes=10)

errors = ModuleValidator.validate(model, strict=False)
errors[-5:]
model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)

def accuracy(preds, labels):
    return (preds == labels).mean()

privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    for images, target in tqdm(train_loader, desc="Train", unit="batch"):
        images = images.to(device)
        target = target.to(device)
        model=model.to(device)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, target)
        loss.backward()

        try:
            optimizer.step()
        except RuntimeError as e:
            print(f"An error occurred during the training: {e}")
            continue

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"Epoch: {epoch}"
        f"\tTrain set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)
            model=model.to(device)
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)

for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    train(model, train_loader, optimizer, epoch + 1, device)

top1_acc = test(model, test_loader, device)
