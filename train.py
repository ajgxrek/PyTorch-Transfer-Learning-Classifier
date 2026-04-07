import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from data_setup import get_dataloaders

# Config
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, class_names = get_dataloaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE)

model = get_model().to(device)

# Only optimize the newly added FC layer; base model stays frozen
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

print(f"Training started on: {device} | Classes: {class_names}")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Eval loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss / len(train_loader):.4f} | Acc: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "resnet18_pets.pth")