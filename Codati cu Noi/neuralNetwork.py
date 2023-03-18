from torch import nn
import torch.nn.functional as F
import torch
from loadData import importData
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torchvision import transforms
class CNN(nn.Module):
    def __init__(self, inChannel=1, numClasses=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=inChannel,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.fc1 = nn.Linear(16*7*7, numClasses)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

data_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

training_dataset = importData(csvFile="./mnist/dataset_training.csv", transforms=transforms.ToTensor())
testing_dataset = importData(csvFile="./mnist/dataset_testing.csv", transforms=transforms.ToTensor())


training_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True)

model =CNN().to(device) #For CNN

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(training_loader)):
        print(batch_idx)
        # Move data to CUDA if is available
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        #data = data.reshape(data.shape[0], -1) #For NN

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent
        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            y = y.to(device=device)

            # Get to correct shape
            #x = x.reshape(x.shape[0], -1) #For NN

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    if loader.dataset.train:
        print("Testing for training dataset")
    else:
        print("Testing for test dataset")

    print(f"Accuracy {num_samples}: {float(num_correct / num_samples) * 100:.2f}")

    model.train()

check_accuracy(training_loader, model)
check_accuracy(testing_loader, model)