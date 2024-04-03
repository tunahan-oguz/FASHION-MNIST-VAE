import torch
import torch.nn as nn


class VAENet(nn.Module):

    class Sampler(nn.Module):

        def forward(self, m, gamma):
            return torch.randn_like(m) * torch.exp(gamma / 2) + m        
    
    def __init__(self, latent_dim : int = 10) -> None:
        super().__init__()

        self.act = nn.ReLU()

        self.input_layer = nn.Linear(784, 256)
        self.encoder_hidden = nn.Linear(256, 100)
        self.mean_layer = nn.Linear(100, latent_dim)
        self.gamma_layer = nn.Linear(100, latent_dim)

        self.sampler = VAENet.Sampler()

        self.initial_decoder = nn.Linear(latent_dim, 100)
        self.hidden_decoder = nn.Linear(100, 256)
        self.final_decoder = nn.Linear(256, 784)

        
    
    def forward(self, x):
        x = self.act(self.input_layer(x))
        x = self.act(self.encoder_hidden(x))
        
        m = self.mean_layer(x)
        gamma = self.gamma_layer(x)

        x = self.sampler(m, gamma)

        x = self.act(self.initial_decoder(x))
        x = self.act(self.hidden_decoder(x))
        x = self.final_decoder(x)
        
        return x, m, gamma
    

    def decode(self, x):

        x = self.act(self.initial_decoder(x))
        x = self.act(self.hidden_decoder(x))
        x = self.final_decoder(x)
        
        return x