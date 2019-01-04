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