import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

bs = 128

train_dataset = datasets.MNIST(root='../data/MNIST', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='../data/MNIST', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim1)  # 784 / 512
        self.fc2 = nn.Linear(h_dim1, h_dim2)  # 512 / 256
        self.fc31 = nn.Linear(h_dim2, z_dim)  # 256 / 2
        self.fc32 = nn.Linear(h_dim2, z_dim)  # 256 / 2

        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim2)  # 2 / 256
        self.fc5 = nn.Linear(h_dim2, h_dim1)  # 256 / 512
        self.fc6 = nn.Linear(h_dim1, x_dim)  # 512 / 784

    def encoder(self, x):
        h = F.relu(self.fc1(x))  # mu
        nn.BatchNorm1d(512)
        h = F.relu(self.fc2(h))  # log_var
        nn.BatchNorm1d(256)
        return self.fc31(h), self.fc32(h)
    # muとlog_varが出力される

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # stdのサイズで作らせる正規分布がtensorをreturnする。
        return eps.mul(std).add_(mu)

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        nn.BatchNorm1d(256)
        h = F.relu(self.fc5(h))
        nn.BatchNorm1d(512)
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))  # muが平均、log_varが標準偏差、mu, log_varは別々にfc31, fc32の出力。
        z = self.sampling(mu, log_var)  # 正規分布でsamplingしたzがdecoderを経るとq_piになる。
        return self.decoder(z), mu, log_var  # decoderで出た結果がq_pi(z)になる。

vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
if torch.cuda.is_available() :
    vae.cuda()

optimizer = optim.Adam(vae.parameters())

def loss_function(recon_x, x, mu, log_var):
    KL_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # -0.5(sigma(1+標準偏差-平均＾２-e^標準偏差）
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    return BCE + KL_divergence
# ここが重要なポイント！

def train(epoch) :
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)

        loss = loss_function(recon_batch,data,mu,log_var)

        loss.backward()
        train_loss = loss.item()
        optimizer.step()

        if batch_idx % 100 == 0 :
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / len(data)))

        if batch_idx * len(data)> 20000 :
            break;

    print('====> Epoch : {} Average loss : {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader :
            data = data.cuda()
            recon, mu, log_var = vae(data)

            test_loss += loss_function(recon, data, mu, log_var).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

z = torch.randn(64,2).cuda()
for epoch in range(1, 201):
    train(epoch)
    test()

    with torch.no_grad():
        sample = vae.decoder(z).cuda()

        save_image(sample.view(64, 1, 28, 28), '../data/VAEsample_{}.png'.format(epoch))


