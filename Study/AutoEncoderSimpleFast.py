import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

batch_size = 100
learning_rate = 0.0002
num_epoch = 1

mnist_train = dset.MNIST("../data/MNIST", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("../data/MNIST", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # batch x 16 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  # batch x 32 x 28 x 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # batch x 64 x 14 x 14
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # batch x 64 x 14 x 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),  # batch x 64 x 7 x 7
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out


encoder = Encoder().cuda()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        out = x.view(batch_size, 256, 7, 7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


if __name__ == '__main__':
    decoder = Decoder().cuda()

    parameters = list(encoder.parameters())+ list(decoder.parameters())
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        encoder, decoder = torch.load('../model/conv_deno_autoencoder.pkl') # pretrain 가중치 저장 파일 불러오기
        print("\n--------model restored--------\n")

    except:
        print("\n--------model not restored--------\n")
        pass

    for i in range(num_epoch):
        for j, [image, label] in enumerate(train_loader):
            noise = init.normal_(torch.FloatTensor(batch_size, 1, 28, 28), 0, 0.1)
            noise = Variable(noise.cuda())

            optimizer.zero_grad()

            image = Variable(image).cuda()
            noise_image = image + noise
            output = encoder(noise_image)
            output = decoder(output)
            loss = loss_func(output, image)

            loss.backward()
            optimizer.step()

            if j % 10 == 0:
                print(loss.data.item())
                torch.save([encoder, decoder], '../model/conv_deno_autoencoder.pkl')

    out_img = torch.squeeze(output.cpu().data)
    print(out_img.size())

    for i in range(5):
        plt.imshow(torch.squeeze(noise_image.cpu().data[i]).numpy(), cmap='gray')
        plt.show()
        plt.imshow(out_img[i].numpy(), cmap='gray')
        plt.show()

    noise = init.normal(torch.FloatTensor(batch_size, 1, 28, 28), 0, 0.1)
    noise = Variable(noise.cuda())

    with torch.no_grad():
        for i in range(1):
            for j, [image, label] in enumerate(test_loader):

                image = Variable(image).cuda()
                noise_image = image + noise
                output = encoder(noise_image)
                output = decoder(output)
                loss = loss_func(output, image)

                if j % 10 == 0:
                    print(loss)

