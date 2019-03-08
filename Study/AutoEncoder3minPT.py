import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

torch.manual_seed(1)

epoch = 10  # epoch
batch_size = 64  # batch size
use_cuda = torch.cuda.is_available()  # GPUを使おうかどうか
device = torch.device("cuda" if use_cuda else "cpu")  # cpuとgpuを区分
print("use device : " , device)

# データを呼ぶ。
trainset = datasets.FashionMNIST(
    root = '../data/FASHIONMNIST',
    train = True,
    transform = transforms.ToTensor(),
    download= True
)
train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2
)


class Autoencoder(nn.Module) :  # アートインコーダを具現してみましょう
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3),
        )  # 四つのレイヤー

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid(),
        )  # 逆にするとOK

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    # 実行する時に自動的に呼び出されます。


if __name__ == '__main__':
    autoencoder = Autoencoder().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    view_data = trainset.train_data[:5].view(-1, 28*28)
    view_data = view_data.type(torch.FloatTensor)/255.  # 見せるためのオリジナルデータ


    def train(autoencoder, train_loader) :
        autoencoder.train()
        for step, (x, label) in enumerate(train_loader):
            x = x.view(-1, 28*28).to(device)
            y = x.view(-1, 28*28).to(device)
            label = label.to(device)

            encoded, decoded = autoencoder(x)

            loss = criterion(decoded, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for curEpoch in range(1, epoch+1):
        train(autoencoder, train_loader)

        test_x = view_data.to(device)
        _, decoded_data = autoencoder(test_x)

        f, a = plt.subplots(2, 5, figsize=(5, 2))
        print("[Epoch {}]".format(curEpoch))
        for i in range(5):
            img = np.reshape(view_data.data.numpy()[i],(28, 28))
            a[0][i].imshow(img, cmap='gray')
            a[0][i].set_xticks(()); a[0][i].set_yticks(())

        for i in range(5):
            img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (28, 28))
            a[1][i].imshow(img, cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.show()


    view_data = trainset.train_data[:1000].view(-1, 28 * 28)
    view_data = view_data.type(torch.FloatTensor) / 255.
    test_x = view_data.to(device)
    encoded_data, _ = autoencoder(test_x)
    encoded_data = encoded_data.to("cpu")

    CLASSES = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    fig = plt.figure(figsize=(10,8))
    ax = Axes3D(fig)

    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()
    Z = encoded_data.data[:, 2].numpy()

    labels = trainset.train_labels[:1000].numpy()

    for x, y, z, s in zip(X, Y, Z, labels):
        name = CLASSES[s]
        color = cm.rainbow(int(255*s/9))
        ax.text(x, y, z, name, backgroundcolor=color)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()
