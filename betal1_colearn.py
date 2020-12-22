import os
from typing import Optional

import torch
import torch.utils.data
from torchvision import transforms, datasets
from torch import nn, optim
from torch.nn import functional as F

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn_examples.training import initial_result, collective_learning_round
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results


class NewPytorchLearner(MachineLearningInterface):
    def __init__(self, model, optimizer, criterion, train_data, test_data, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_data
        self.test_loader = test_data
        self.device = device
        self.vote_score = self.test(self.train_loader)

    def get_current_weights(self) -> Weights:
        w = Weights(weights=[x.clone() for x in self.model.parameters()])
        return w

    def set_weights(self, weights: Weights):
        with torch.no_grad():
            for new_param, old_param in zip(weights.weights,
                                            self.model.parameters()):
                old_param.set_(new_param)

    def train(self):
        self.model.train()

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            labels = labels.to(self.device)
            output = self.model(data)
            # do something with loss
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

    def propose_weights(self) -> Weights:
        current_weights = self.get_current_weights()
        self.train()
        new_weights = self.get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def test_weights(self, weights: Weights, eval_config: Optional[dict] = None) -> ProposedWeights:
        current_weights = self.get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.train_loader)
        if self.test_loader:
            test_score = self.test(self.test_loader)
        else:
            test_score = 0
        vote = self.vote(vote_score)

        self.set_weights(current_weights)
        return ProposedWeights(weights=weights,
                               vote_accuracy=vote_score,
                               test_accuracy=test_score,
                               vote=vote
                               )

    def vote(self, new_score):
        return new_score > self.vote_score

    def test(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, labels)
                total_loss += loss
        return total_loss

    def accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.train_loader)


class BetaL1(MachineLearningInterface):
    def __init__(self, seed, batch_size):
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

        torch.manual_seed(seed)
        no_cuda = False
        cuda = not no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        train_root = 'Trainpatches'
        # val_root = 'Testpatches'
        # val_root = 'Testpatches'
        val_root = train_root
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(16, fill=0, padding_mode='constant'),
            transforms.ToTensor()])

        "DATA LOADER"
        # train_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(train_root, transform=transform),
        #     batch_size=args.batch_size, shuffle=True, **kwargs)
        # test_loader = torch.utils.data.DataLoader(
        #     datasets.ImageFolder(val_root, transform=transform),
        #     batch_size=args.batch_size, shuffle=False, **kwargs)
        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(train_root, transform=transform, download=True),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(val_root, transform=transform, download=True),
            batch_size=batch_size, shuffle=False, **kwargs)

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
        self.model = BetaVAE().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", min_lr=1e-6, patience=2, verbose=2)
        val_losses = []
        train_losses = []
        self.beta = DeterministicWarmup(n_steps=N_WARM_UP, t_max=1)  # BETA paramter created in the DeterminsticWarmup class
        self.epoch = 1

    def get_current_weights(self) -> Weights:
        w = Weights(weights=[x.clone() for x in self.model.parameters()])
        return w

    def propose_weights(self) -> Weights:
        if self.epoch == 1:
            _beta = 0
        else:
            _beta = next(self.beta)
        if _beta < 1:
            print("Beta: ", _beta)
        self.model.train()
        train_loss = 0
        num_batches = 0

        for batch_idx, (data, label) in enumerate(self.train_loader):
            if batch_idx > 10:  # training is slooooow
                break

            "WITH BETA"
            images = data
            images = images.to(self.device)
            reconstruction, mu, logvar = self.model(images)
            # np.save("results_" + NAME + "/numpy/mu_train_" + str(num_batches) + ".npy", mu.detach().cpu().numpy())
            # np.save("results_" + NAME + "/numpy/sigma_train_" + str(num_batches) + ".npy", logvar.detach().cpu().numpy())
            # np.save("results_" + NAME + "/numpy/label_train_" + str(num_batches) + ".npy", label.detach().cpu().numpy())
            self.optimizer.zero_grad()

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
            self.optimizer.step()

            # saving example reconstruction per epoch
            # if batch_idx == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n].to(self.device),
            #                             reconstruction.view(args.batch_size, 3, 64, 64)[:n]])
            #     save_image(comparison.cpu().detach(),
            #                'results_' + NAME + '/reconstruction_train_' + str(epoch) + '.png', nrow=n)

            num_batches += 1

        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #     self.epoch, train_loss / len(self.train_loader.dataset)))
        # if self.epoch > 0:
        #     self.train_losses.append(train_loss / len(self.train_loader.dataset))
        # train_losses_np = train_losses.numpy()
        # np.save(NAME + "_trainloss.npy", train_losses)
        # return train_loss, last_train_step + num_batches
        self.epoch += 1
        return self.get_current_weights()

    def _set_weights(self, weights: Weights):
        with torch.no_grad():
            for new_param, old_param in zip(weights.weights,
                                            self.model.parameters()):
                old_param.set_(new_param)

    def test_weights(self, weights: Weights, eval_config: Optional[dict] = None) -> ProposedWeights:
        if weights is None:
            weights = self.get_current_weights()

        self.model.eval()
        test_loss = 0
        num_batches = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(self.test_loader):
                if i > 10:  # testing is also slooooow
                    break

                "WITH BETA"
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                # np.save("results_" + NAME + "/numpy/mu" + str(num_batches) + ".npy", mu.detach().cpu().numpy())
                # np.save("results_" + NAME + "/numpy/sigma" + str(num_batches) + ".npy", logvar.detach().cpu().numpy())
                # np.save("results_" + NAME + "/numpy/label" + str(num_batches) + ".npy", label.detach().cpu().numpy())
                # munp = mu.detach().cpu().numpy()
                # lognp = logvar.detach().cpu().numpy()
                # labelnp = label.detach().cpu().numpy()

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

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        # if epoch > 0:
        #     val_losses.append(test_loss)  # adjust to see the minimum loss, removing the first n-epochs
        # np.save(NAME + "_val_loss.npy", val_losses)

        return ProposedWeights(weights=weights,
                               vote_accuracy=test_loss,
                               test_accuracy=0,
                               vote=True)

    def accept_weights(self, weights: Weights):
        self._set_weights(weights)
        # pw = self.test_weights()
        # self.score =
        # pass


if __name__ == "__main__":
    n_learners = 5
    batch_size = 32
    seed = 42
    n_epochs = 20
    make_plot = True
    vote_threshold = 0.5
    all_learner_models = [BetaL1(seed=42, batch_size=batch_size) for _ in range(n_learners)]

    results = Results()
    # Get initial accuracy
    results.data.append(initial_result(all_learner_models))

    for i in range(n_epochs):
        results.data.append(
            collective_learning_round(all_learner_models,
                                      vote_threshold, i)
        )

        if make_plot:
            # then make an updating graph
            plot_results(results, n_learners, block=False)
            plot_votes(results, block=False)

    if make_plot:
        plot_results(results, n_learners, block=False)
        plot_votes(results, block=True)
