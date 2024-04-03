import torch
from model import VAENet
import numpy as np
import matplotlib.pyplot as plt

def get_codings(n_samples, latent_dim = 10):
    return torch.randn((n_samples, latent_dim))


codes = get_codings(5)
net = VAENet()
st_dict = torch.load("model.pth")
net.load_state_dict(st_dict)

with torch.no_grad():
    imgs = net.decode(codes)
    imgs = imgs.reshape(-1, 28, 28)

for img in imgs:
    img = img.numpy()
    img = (img * 255).astype(np.uint8)
    plt.imshow(img, cmap="gray")
    plt.show()