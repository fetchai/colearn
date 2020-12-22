from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
# from torch.autograd import Variable
import matplotlib.pyplot as plt
# import seaborn as sns
import os
# import torch.nn.init as init
import numpy as np

# Author: Isabella Douzoglou
# Partial code from:
# https://github.com/pytorch/examples/blob/master/vae/main.py
# https://github.com/TDehaene/blogposts/blob/master/vae_new_food/notebooks/vae_pytorch.ipynb
# https://github.com/federicobergamin/Ladder-Variational-Autoencoders/

"HYPERPARAMETERS"
TITLE = "BetaL1_"
EPOCH = 2
N_WARM_UP = 100
LEARNING_RATE = 0.0001  # 1e-4 #(MINE)
LOSSTYPE = "BETA"
latent_dim = 20
calc_shape = 256
pixel = 32
dim = pixel + pixel
NAME = TITLE + "LR:" + str(LEARNING_RATE) + "_WU:" + str(N_WARM_UP) + "_E:" + str(EPOCH) + "_Ldim:" + str(latent_dim)

# "RESULTS FOLDER"
# try:
#     os.mkdir("results_" + NAME + "/")
#     os.mkdir("results_" + NAME + "/numpy/")
# except OSError:
#     print("results folder exist, deleting re-creating")
#     os.system("rm -r results_" + NAME + "/")
#     os.mkdir("results_" + NAME + "/")
#     os.mkdir("results_" + NAME + "/numpy/")

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=EPOCH, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_root = 'Trainpatches'
# val_root = 'Testpatches'
val_root = train_root
transform = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Pad(16, fill=0, padding_mode='constant'),
    transforms.ToTensor()])
# Isabella's brain images are 3x64x64, and CIFAR10 is 3x32x32, hence why I've added the Pad transform

"DATA LOADER"
# train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(train_root, transform=transform),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(val_root, transform=transform),
#     batch_size=args.batch_size, shuffle=False, **kwargs)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(train_root, transform=transform, download=True),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(val_root, transform=transform, download=True),
    batch_size=args.batch_size, shuffle=False, **kwargs)

"DECLARE MODEL CLASS"


class BetaVAE(nn.Module):
    def __init__(self):
        super(BetaVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 256, 4, 1)

        self.fc11 = nn.Linear(calc_shape, latent_dim)
        self.fc12 = nn.Linear(calc_shape, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 256)

        self.deconv1 = nn.ConvTranspose2d(256, 64, 4)
        self.deconv2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def encode(self, x):
        x = self.conv1(x)  # B,  32, 32, 32
        x = F.relu(x)
        x = self.conv2(x)  # B,  32, 16, 16
        x = F.relu(x)
        x = self.conv3(x)  # B,  64,  8,  8
        x = F.relu(x)
        x = self.conv4(x)  # B,  64,  4,  4
        x = F.relu(x)
        x = self.conv5(x)  # B, 256,  1,  1
        x = F.relu(x)

        flat = x.view((-1, 256 * 1 * 1))  # B, 256
        return self.fc11(flat), self.fc12(flat)

    @staticmethod
    def reparameterize(mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc2(z)
        unflat = z.view(-1, 256, 1, 1)
        aggregated = unflat
        x = F.relu(aggregated)
        x = self.deconv1(x)  # B,  64,  4,  4
        x = F.relu(x)
        x = self.deconv2(x)  # B,  64,  8,  8
        x = F.relu(x)
        x = F.relu(x)
        x = self.deconv3(x)  # B,  32, 16, 16
        x = F.relu(x)
        x = self.deconv4(x)  # B,  32, 32, 32
        x = F.relu(x)
        x = self.deconv5(x)  # B, nc, 64, 64
        return torch.sigmoid(x)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        decoded = self.decode(z)
        return decoded, mu, log_sigma


# def reparametrize(mu, logvar):
#     std = logvar.div(2).exp()
#     eps = Variable(std.data.new(std.size()).normal_())
#     return mu + std * eps


class DeterministicWarmup(object):
    def __init__(self, n_steps, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.increase = self.t_max / n_steps

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.increase
        self.t = self.t_max if t > self.t_max else t
        return self.t


"DECLARE NEW MODEL"
model = BetaVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", min_lr=1e-6, patience=2, verbose=2)
val_losses = []
train_losses = []
beta = DeterministicWarmup(n_steps=N_WARM_UP, t_max=1)  # BETA paramter created in the DeterminsticWarmup class

"TRAIN"


def train(epoch, last_train_step=0):
    if epoch == 1:
        _beta = 0
    else:
        _beta = next(beta)
    if _beta < 1:
        print("Beta: ", _beta)
    model.train()
    train_loss = 0
    num_batches = 0

    for batch_idx, (data, label) in enumerate(train_loader):
        if batch_idx > 10:  # training is slooooow
            break

        "WITH BETA"
        images = data
        images = images.to(device)
        reconstruction, mu, logvar = model(images)
        # np.save("results_" + NAME + "/numpy/mu_train_" + str(num_batches) + ".npy", mu.detach().cpu().numpy())
        # np.save("results_" + NAME + "/numpy/sigma_train_" + str(num_batches) + ".npy", logvar.detach().cpu().numpy())
        # np.save("results_" + NAME + "/numpy/label_train_" + str(num_batches) + ".npy", label.detach().cpu().numpy())
        optimizer.zero_grad()

        # Original: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Adjustement: L1 Loss + Beta

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        likelihood = - F.l1_loss(reconstruction, images, reduction='sum')
        elbo = likelihood - _beta * torch.sum(KLD)
        loss = - elbo / len(images)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # saving example reconstruction per epoch
        # if batch_idx == 0:
        #     n = min(data.size(0), 8)
        #     comparison = torch.cat([data[:n].to(device),
        #                             reconstruction.view(args.batch_size, 3, 64, 64)[:n]])
        #     save_image(comparison.cpu().detach(),
        #                'results_' + NAME + '/reconstruction_train_' + str(epoch) + '.png', nrow=n)

        num_batches += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    if epoch > 0:
        train_losses.append(train_loss / len(train_loader.dataset))
    # train_losses_np = train_losses.numpy()
    # np.save(NAME + "_trainloss.npy", train_losses)
    return train_loss, last_train_step + num_batches


"TEST"


def test(epoch):
    model.eval()
    test_loss = 0
    num_batches = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if i > 10:  # testing is also slooooow
                break

            "WITH BETA"
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            # np.save("results_" + NAME + "/numpy/mu" + str(num_batches) + ".npy", mu.detach().cpu().numpy())
            # np.save("results_" + NAME + "/numpy/sigma" + str(num_batches) + ".npy", logvar.detach().cpu().numpy())
            # np.save("results_" + NAME + "/numpy/label" + str(num_batches) + ".npy", label.detach().cpu().numpy())
            munp = mu.detach().cpu().numpy()
            lognp = logvar.detach().cpu().numpy()
            labelnp = label.detach().cpu().numpy()

            # Original: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # Adjustement: L1 Loss 

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            likelihood = - F.l1_loss(recon_batch, data, reduction='sum')
            elbo = likelihood - torch.sum(KLD)
            loss = - elbo / len(data)
            test_loss += loss.item()
            num_batches += 1

            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                             recon_batch.view(args.batch_size, 3, 64, 64)[:n]])
            #     save_image(comparison.cpu(),
            #                'results_' + NAME + '/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    if epoch > 0:
        val_losses.append(test_loss)  # adjust to see the minimum loss, removing the first n-epochs
    # np.save(NAME + "_val_loss.npy", val_losses)


if __name__ == "__main__":
    last_train_step = 0
    result_label = np.array([])
    for epoch in range(1, args.epochs + 1):
        "TRAIN"
        train_loss, last_train_step = train(epoch, last_train_step)
        "TEST"
        test(epoch)
        "SAVE MODEL"
        # torch.save(model.state_dict(), NAME + ".pt")

        "LOSS PLOT"
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(range(len(val_losses)), val_losses)
        plt.title("Validation loss and loss per epoch", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
        plt.savefig("LOSS_" + NAME + ".png")
        plt.close()

        with torch.no_grad():
            sample = torch.randn(args.batch_size, latent_dim).to(device)
            sample = model.decode(sample).cpu()
            # save_image(sample.view(args.batch_size, 3, 64, 64),
            #            'results_' + NAME + '/sample_' + str(epoch) + '.png')
