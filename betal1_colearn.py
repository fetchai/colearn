from typing import Optional

import torch
import torch.utils.data
from torchvision import transforms, datasets
from torch import nn
from torch.nn import functional

from colearn.ml_interface import MachineLearningInterface, Weights, ProposedWeights
from colearn_examples.training import initial_result, collective_learning_round
from colearn_examples.utils.plot import plot_results, plot_votes
from colearn_examples.utils.results import Results


class NewPytorchLearner(MachineLearningInterface):
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 device=torch.device("cpu"),
                 criterion=None,
                 minimise_criterion=True):
        self.model: torch.nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion = criterion
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.test_loader: Optional[torch.utils.data.DataLoader] = test_loader
        self.device = device
        self.vote_score = self.test(self.train_loader)
        self.minimise_criterion = minimise_criterion

    def mli_get_current_weights(self) -> Weights:
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

    def mli_propose_weights(self) -> Weights:
        current_weights = self.mli_get_current_weights()
        self.train()
        new_weights = self.mli_get_current_weights()
        self.set_weights(current_weights)
        return new_weights

    def mli_test_weights(self, weights: Weights, eval_config: Optional[dict] = None) -> ProposedWeights:
        current_weights = self.mli_get_current_weights()
        self.set_weights(weights)

        vote_score = self.test(self.train_loader)
        print(vote_score)
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

    def vote(self, new_score) -> bool:
        if self.minimise_criterion:
            return new_score <= self.vote_score
        else:
            return new_score >= self.vote_score

    def test(self, loader) -> float:
        if not self.criterion:
            raise Exception("Criterion is unspecified so test method cannot be used")

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, labels)
                total_loss += loss
        return float(total_loss / len(loader))

    def mli_accept_weights(self, weights: Weights):
        self.set_weights(weights)
        self.vote_score = self.test(self.train_loader)


class BetaL1(NewPytorchLearner):
    def __init__(self, seed, train_loader, test_loader,
                 device=torch.device("cpu")):
        # HYPERPARAMETERS
        n_warm_up = 100
        learning_rate = 0.00001  # 1e-4 #(MINE)
        latent_dim = 20
        calc_shape = 256

        torch.manual_seed(seed)
        self.device = device

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
                x = functional.relu(x)
                x = self.conv2(x)  # B,  32, 16, 16
                x = functional.relu(x)
                x = self.conv3(x)  # B,  64,  8,  8
                x = functional.relu(x)
                x = self.conv4(x)  # B,  64,  4,  4
                x = functional.relu(x)
                x = self.conv5(x)  # B, 256,  1,  1
                x = functional.relu(x)

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
                x = functional.relu(aggregated)
                x = self.deconv1(x)  # B,  64,  4,  4
                x = functional.relu(x)
                x = self.deconv2(x)  # B,  64,  8,  8
                x = functional.relu(x)
                x = functional.relu(x)
                x = self.deconv3(x)  # B,  32, 16, 16
                x = functional.relu(x)
                x = self.deconv4(x)  # B,  32, 32, 32
                x = functional.relu(x)
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

        # DECLARE NEW MODEL
        model = BetaVAE().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = DeterministicWarmup(n_steps=n_warm_up,
                                        t_max=1)  # BETA paramter created in the DeterminsticWarmup class
        self.epoch = 1

        super().__init__(model, optimizer, train_loader, test_loader, self.device)

    def vote(self, new_score: float):
        # return new_score < self.vote_score
        return True

    def train(self):
        if self.epoch == 1:
            _beta = 0
        else:
            _beta = next(self.beta)
        if _beta < 1:
            print("Beta: ", _beta)
        self.model.train()
        train_loss = 0

        for batch_idx, (data, label) in enumerate(self.train_loader):
            if batch_idx > 50:  # training is slooooow
                break

            images = data
            images = images.to(self.device)
            reconstruction, mu, logvar = self.model(images)
            self.optimizer.zero_grad()

            # Original: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # Adjustement: L1 Loss + Beta

            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            likelihood = - functional.l1_loss(reconstruction, images, reduction='sum')
            elbo = likelihood - _beta * torch.sum(kld)
            loss = - elbo / len(images)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        self.epoch += 1
        return train_loss

    def test(self, loader):
        self.model.eval()
        test_loss = 0
        num_samples = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(loader):
                if i > 50:  # testing is also slooooow
                    break

                data: torch.Tensor = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)

                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                likelihood = - functional.l1_loss(recon_batch, data, reduction='sum')
                elbo = likelihood - torch.sum(kld)
                loss = - elbo / len(data)
                test_loss += loss.item()
                num_samples += data.shape[0]

        test_loss /= num_samples
        print('====> Test set loss: {:.4f}'.format(test_loss))

        return test_loss


if __name__ == "__main__":
    n_learners = 5
    batch_size = 32
    seed = 42
    n_epochs = 20
    make_plot = True
    vote_threshold = 0.5
    no_cuda = False
    train_fraction = 0.9

    cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # move data setup outside
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_root = 'Trainpatches'
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(16, fill=0, padding_mode='constant'),
        transforms.ToTensor()])

    # DATA LOADER
    data = datasets.CIFAR10(train_root, transform=transform, download=True)
    n_train = int(train_fraction * len(data))
    n_test = len(data) - n_train
    train_data, test_data = torch.utils.data.random_split(data, [n_train, n_test])

    data_split = [len(train_data)//n_learners] * n_learners
    learner_train_data = torch.utils.data.random_split(train_data, data_split)
    learner_train_dataloaders = [torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size, shuffle=True, **kwargs) for ds in learner_train_data]

    data_split = [len(test_data)//n_learners] * n_learners
    learner_test_data = torch.utils.data.random_split(test_data, data_split)
    learner_test_dataloaders = [torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size, shuffle=True, **kwargs) for ds in learner_test_data]

    all_learner_models = []
    for i in range(n_learners):
        all_learner_models.append(BetaL1(seed=42,
                                         train_loader=learner_train_dataloaders[i],
                                         test_loader=learner_test_dataloaders[i],
                                         device=device
                                         ))

    results = Results()
    # Get initial accuracy
    results.data.append(initial_result(all_learner_models))

    for epoch in range(n_epochs):
        results.data.append(
            collective_learning_round(all_learner_models,
                                      vote_threshold, epoch)
        )

        if make_plot:
            # then make an updating graph
            plot_results(results, n_learners, block=False)
            plot_votes(results, block=False)

    if make_plot:
        plot_results(results, n_learners, block=False)
        plot_votes(results, block=True)
