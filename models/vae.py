import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, feature_size=257*2, latent_size=128):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.feature_size = feature_size

        # encode
        self.fc1  = nn.Linear(feature_size, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size, 512)
        self.fc4 = nn.Linear(512, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):  
        '''
        x: (bs, feature_size)
        '''
        h1 = self.elu(self.fc1(x))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode2(self, x):
        mu, logvar = self.encode(x.view(-1, self.feature_size))
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, condition_size)
        '''
        h3 = self.elu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.feature_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar