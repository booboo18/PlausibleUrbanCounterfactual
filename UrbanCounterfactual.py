
"""
"Explaining Holistic Image Regressors and Classifiers in Urban Analytics with Plausible Counterfactuals"  in the International Journal of Geographical Information Science. 

We propose a new form of plausible counterfactual explanation designed to explain the behaviour of computer vision systems 
used in urban analytics that make predictions based on properties across the entire image,
rather than specific regions of it. This is the code to run the urban counterfactual analysis. 
As the data cannot be redistributed due to commercial license, simulated data and model have been used instead. 
Users would need to replace the VAE model and regressor with their own data and model in order to run the counterfactual algorithm. 

Authors: Stephen Law, Rikuo Hasegawa, Brooks Paige, Chris Russell, Andrew Elliott
License: MIT
"""

# code starts here

from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from numpy.random import rand
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# functions for the counterfactual explanations
def lagrangian(res, tgt):
    return (res - tgt) ** 2

def counterfactual_regressor_only(
    x,
    y,
    target,
    regressor,
    new_L=False,
    lambda_target=1,
    lr=1,
    iterations=5000,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    y = np.asarray(y)
    y = np.expand_dims(y, axis=1)
    target = np.array(target)
    target = Variable(torch.from_numpy(target), requires_grad=False).float().to(device)

    curr_L = Variable(torch.from_numpy(x), requires_grad=False).to(device)
    if new_L is False:
        new_L = Variable(
            torch.randn(curr_L.shape).double().to(device) / 1.0e5 + curr_L.double(),
            requires_grad=True,
        )
    else:
        new_L = Variable(new_L, requires_grad=True)
    new_L.to(device)

    l1 = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.ASGD(
        [
            new_L,
        ],
        lr=lr,
        t0=100,
        weight_decay=1e-5,
    )

    imgs = Variable(
        torch.empty([x.shape[0], 3, 244, 244], device=device, requires_grad=False),
        requires_grad=False,
    )
    distance = torch.empty([1], device=device, requires_grad=False)
    target_loss = torch.empty([1], device=device, requires_grad=False)
    loss = torch.empty([1], device=device, requires_grad=False)
    regression_result = torch.empty_like(target, device=device, requires_grad=False)

    optimizer.zero_grad(True)

    for i in tqdm(range(iterations)):
        regression_result = regressor(new_L.float())
        target_loss = lambda_target * (
            lagrangian(tgt=target, res=regression_result).mean()
        )
        distance = l1(new_L.double(), curr_L.double())
        loss = distance + target_loss
        optimizer.zero_grad(True)
        loss.backward(retain_graph=True)
        optimizer.step()

    return new_L

def defineVAEModel1():
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                nn.Flatten(),
            )
            self.mu = nn.Linear(64 * 14 * 14, 1568)
            self.var = nn.Linear(64 * 14 * 14, 1568)

        def forward(self, x):
            x = self.encoder(x)
            z_mu = self.mu(x)
            z_var = self.var(x)
            return z_mu, z_var

    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.latent_to_hidden = nn.Linear(1568, 64 * 14 * 14)  # + n_classes
            self.decoder = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    64, 64, kernel_size=3, stride=1, padding=0
                ), 
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    64, 64, kernel_size=3, stride=1, padding=0
                ),  
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0),
            )  

        def forward(self, x):
            x = self.latent_to_hidden(x)
            x = x.view(-1, 64, 14, 14)
            x = self.decoder(x)
            return x

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()  
            self.decoder = Decoder()  
        def forward(self, x):
            z_mu, z_var = self.encoder(x)
            std = torch.exp(z_var / 2)
            eps = torch.randn_like(std)
            x_sample = eps.mul(std).add_(z_mu)
            predicted = self.decoder(x_sample)
            return predicted, z_mu, z_var

    device = torch.device("cpu")
    model = VAE().to(device)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.discriminator = nn.Sequential(
                nn.Conv2d(3, 28, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(28, 36, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(36, 48, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(128),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(48, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.InstanceNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                nn.Flatten(),
                nn.Linear((64 * 7 * 7), (1), bias=True),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.discriminator(x)
            return x

    discriminator = Discriminator()  

    return model, discriminator  

# main script starts here
lat_dim=1568

# this is a simulated input and target files. please replace this with your own files in this shape.
X = np.load('testX.npy') # this is a simulated input that we created for demonstration
y = np.load('testy.npy') # this is a simulated target that we created for demonstration
target=2.5 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# this setup and loads the regression model for the test data 
regression_model = nn.Sequential( 
  nn.Linear(lat_dim, 1),
  ).to(device)

# user needs to slot in their own regressor pth file here to load.
#path = "test_REG_model.pth" 
#regression_model.load_state_dict(torch.load(path))

regression_model.requires_grad = False
regression_model.eval()

# this setup and loads the VAEGAN model for the test data
model, discriminator = defineVAEModel1() 

## user needs to slot in their own vae model pth file here to load.
#path = "test_VAE_model.pth" 
#model.load_state_dict(torch.load(path, map_location=device))

model.to(device)
model.requires_grad = False
model.eval()
decoder = model.decoder

# get counterfactuals using the regression model
counterfactuals = counterfactual_regressor_only(
            X,
            y,
            target,
            regression_model,
            new_L=False,
            lambda_target=100,
            lr=0.001,
            iterations=20_000,
            )

# reconstruct counterfactuals
img = decoder.float()(counterfactuals.float())
recon=img.cpu().detach().numpy().squeeze().transpose(1,2,0)
plt.imshow(recon) # random outputs