import torch
from torch.autograd import Variable
from datetime import datetime
start = datetime.now()

N,D = 3,4

x = Variable(torch.randn(N,D),requires_grad=True)
y = Variable(torch.randn(N,D),requires_grad=True)
z = Variable(torch.randn(N,D),requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward(gradient=torch.FloatTensor([1.0]))

print(x.grad)
print(y.grad)
print(z.grad)
print(datetime.now()-start)