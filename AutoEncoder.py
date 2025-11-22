import torch
import torch.nn as nn

'''
Autoencoder on MNIST dataset (28x28 images)
'''

class AutoEncoder_Linear(nn.Module):
    def __init__(self, inupt_size, latent_size=64):
        super().__init__()
        self.input_size = inupt_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size * self.input_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.input_size * self.input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        # print(f'latent.shape: {latent.shape}')
        decoded_output = self.decoder(latent)
        reconstructed = decoded_output.view(-1, 1, self.input_size, self.input_size)
        return reconstructed
    


        