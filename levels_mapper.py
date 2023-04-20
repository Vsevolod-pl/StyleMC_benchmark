import torch
from torch.nn import Module
from StyleGAN import EqualLinear, PixelNorm

def Mapper(latent_dim=512, n_layers=4):
    layers = [PixelNorm()]
    for i in range(n_layers):
        layers.append(
            EqualLinear(
                latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
            )
        )

    return torch.nn.Sequential(*layers)

class LevelsMapper(Module):

    def __init__(self, latent_dim=512):
        super(LevelsMapper, self).__init__()

        self.coarse_mapping = Mapper(latent_dim=latent_dim, n_layers=4)
        self.medium_mapping = Mapper(latent_dim=latent_dim, n_layers=4)
        self.fine_mapping = Mapper(latent_dim=latent_dim, n_layers=4)

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        x_coarse = self.coarse_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out

def load_mapper(path, device='cpu'):
    mapper = LevelsMapper().to(device)
    mapper.load_state_dict(torch.load(path))
    return mapper