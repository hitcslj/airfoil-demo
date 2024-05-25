import torch
from torch import nn


class CVAE(nn.Module):
    def __init__(self, feature_size=257*2, latent_size=20, condition_size=37*2):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.condition_size = condition_size
        self.latent_size = latent_size

        # encode
        self.fc1  = nn.Linear(feature_size + condition_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + condition_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+condition_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, condition_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+condition_size)
        h3 = self.elu(self.fc3(inputs))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.feature_size), c.view(-1,self.condition_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c.view(-1,self.condition_size)), mu, logvar

    def sample(self,c):
        batch = c.shape[0]
        z = torch.randn((batch,self.latent_size)).to(c.device)
        recons_batch = self.decode(z, c.view(-1,self.condition_size))
        return recons_batch.reshape(-1,257,2)