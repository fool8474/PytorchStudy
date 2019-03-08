import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w*x+b
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)

# ----------------------------

x = torch.randn(10,3)
y = torch.randn(10,2)

linear = nn.Linear(3,2)
print('w: ', linear.weight)
print('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)
loss = criterion(pred, y)
print('loss : ', loss.item())

loss.backward()

print("dL/dw: ", linear.weight.grad)
print("dL/db: ", linear.bias.grad)

optimizer.step()

pred = linear(x)
loss = criterion(pred,y)
print('loss after 1 step opt : ', loss.item())

print("------------------------")
x = np.array([[1,2],[3,4]])
y = torch.from_numpy(x)
z = y.numpy()

print(x)
print(y)
print(z)

print("------------------------")
resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

images = torch.randn(64,3,224,224)
outputs = resnet(images)
print(outputs.size())


print("------------------------")

class NeuralNet(nn.Module) :
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x) :
        out = self.fc1(x)
        out = self.relu(x)
        out = self.fc2(x)
        out = self.relu2(x)
        out = self.fc3(x)
        return out

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../MNIST/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one MNIST pair (read MNIST from disk).
image, label = train_dataset[0]
print(image.size())
print(label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# When iteration starts, queue and thread start to load MNIST from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the MNIST loader is as below.
for images, labels in train_loader:
    pass

