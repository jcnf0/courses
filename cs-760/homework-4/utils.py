import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

class Net(nn.Module):
    def __init__(self, d1 = 200):
        super(Net, self).__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(784, d1)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(d1, 10)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_loss(y, y_hat):
    return -np.sum(y * np.log(y_hat))

def training_scratch(d1 = 200, learning_rate = 0.01, num_epochs = 10, batch_size = 64):
    def forward(x, W1, W2):
        z1 = np.dot(W1, x)
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1)
        y_hat = softmax(z2)
        return z1, a1, z2, y_hat

    def backward(x, y, y_hat, a1, W2):
        # print(f"SHAPES: x {x.shape}, y {y}, y_hat {y_hat.shape}, a1 {a1.shape}, W2 {W2.shape}")
        a1 = a1.reshape(-1, 1)
        x = x.reshape(-1, 1)
        delta2 = (y_hat - y).reshape(-1, 1)
        grad_W2 = np.dot(delta2, a1.reshape(1, -1))
        delta1 = np.dot(W2.T, delta2) * (a1*(1 - a1))
        grad_W1 = np.dot(delta1, x.T)
        return grad_W1, grad_W2

    def train(X, Y, W1, W2):
        losses = []
        num_batches = X.shape[0] // batch_size
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                X_batch = X[start_idx:end_idx]
                Y_batch = Y[start_idx:end_idx]
                X_batch = X_batch.reshape(batch_size, -1)
                grad_W1, grad_W2 = 0, 0
                for j in range(batch_size):
                    x_batch = X_batch[j]
                    y_batch = np.zeros(10)
                    y_batch[Y_batch[j]] = 1
                    _, a1, _, y_hat = forward(x_batch, W1, W2)
                    epoch_loss += cross_entropy_loss(y_batch, y_hat)
                    newgrad_W1, newgrad_W2 = backward(x_batch, y_batch, y_hat, a1, W2)
                    grad_W1 += newgrad_W1
                    grad_W2 += newgrad_W2
                W1 -= learning_rate * grad_W1
                W2 -= learning_rate * grad_W2

            epoch_loss /= num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')
            losses.append(epoch_loss)
        return losses

    def predict(X, W1, W2):
        y_pred = []
        for i in range(X.shape[0]):
            x = X[i].reshape(-1, 1)
            _, _, _, y_hat = forward(x, W1, W2)
            y_pred.append(np.argmax(y_hat))
        return np.array(y_pred)

    def accuracy(y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_true)

    # Load MNIST dataset with torchvision
    train_ds = datasets.MNIST(root="./data", train=True, download=True)
    test_ds = datasets.MNIST(root="./data", train=False, download=True)
    # Extract features and labels
    X_train = train_ds.data.numpy()
    y_train = train_ds.targets.numpy()
    X_train = X_train / 255.0
    y_train = y_train.astype(np.uint8)

    X_test = test_ds.data.numpy()
    y_test = test_ds.targets.numpy()
    X_test = X_test / 255.0
    y_test = y_test.astype(np.uint8)

    # Initialize weights
    W1 = np.random.randn(d1, 784)
    W1 = (W1 - np.min(W1))*2/(np.max(W1) - np.min(W1)) - 1
    W2 = np.random.randn(10, d1)
    W2 = (W2 - np.min(W2))*2/(np.max(W2) - np.min(W2)) - 1

    # Train the model
    losses = train(X_train, y_train, W1, W2)

    # Plot the learning curve
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Scratch D1 = {}, Batch Size = {}, LR = {}, Weight Init = random".format(d1,batch_size,learning_rate))
    plt.savefig("figures/Scratch_Loss_D1_{}_BatchSize_{}_LR_{}.pdf".format(d1,batch_size,learning_rate))
    plt.clf()

    # Predict the labels
    y_pred = predict(X_test, W1, W2)

    # Calculate the accuracy
    print(f"Accuracy: {100*accuracy(y_pred, y_test)}%")
    return 100*accuracy(y_pred, y_test)
    
def training_pytorch(d1 = 200, learning_rate = 0.01, num_epochs = 10, batch_size = 64, weigth_init = "random"):
    loss_values = []
    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = Net(d1)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Initialize the weights
    if weigth_init == "zeros":
        model.fc1.weight.data = torch.zeros(d1, 784)
        model.fc2.weight.data = torch.zeros(10, d1)
    elif weigth_init == "random":
        model.fc1.weight.data = torch.randn(d1, 784)
        model.fc2.weight.data = torch.randn(10, d1)
        # Set between -1 and 1
        model.fc1.weight.data = (model.fc1.weight.data - torch.min(model.fc1.weight.data))*2/(torch.max(model.fc1.weight.data) - torch.min(model.fc1.weight.data)) - 1
        model.fc2.weight.data = (model.fc2.weight.data - torch.min(model.fc2.weight.data))*2/(torch.max(model.fc2.weight.data) - torch.min(model.fc2.weight.data)) - 1

    # Training the model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        loss_values.append(loss.item())

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total}%')

    # Plot the learning curve
    plt.plot(loss_values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pytorch D1 = {}, Batch Size = {}, LR = {}, Weigth Init = {}".format(d1,batch_size,learning_rate,weigth_init))

    if os.path.isdir("figures") == False:
        os.mkdir("figures")

    plt.savefig("figures/Loss_D1_{}_BatchSize_{}_LR_{}_WeigthInit_{}".format(d1,batch_size,learning_rate, weigth_init)+ ".pdf")
    plt.clf()
    return 100 * correct / total