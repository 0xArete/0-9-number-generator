import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        """Reshape to another dimensnsion"""
        self.shape = shape, # <- create tuple
    
    def forward(self, x):
        return x.reshape(*self.shape)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Create structure of NN
        """
        # Only for convinienct in main.py
        self.input_size = 100
        # Neural Network layers
        self.model = nn.Sequential(
            # reshape to z (_, z, y, x)
            View((1, 100, 1, 1)),
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),

            View((28, 28))
            )

        # No loss function; will use one from discriminator to calculate error

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Timestap of loss for ploting progress
        self.progress = []
        self.counter = 0

    def forward(self, inputs):
        """
        Pass through NN and get its answer
        """
        return self.model(inputs)

    def train(self, D, inputs, targets):
        """
        Train NN; Take tensor of image with label identificator of image;
        Pass through NN; get loss/cost function and backpropagate NN
        to tweak weights (layers)
        """
        g_output = self.forward(inputs)
        d_outputs = D.forward(g_output)

        loss = D.loss_function(d_outputs, targets)

        # Backpropagation
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.counter += 1
        if self.counter % 100 == 0:
            self.progress.append(loss.item())

    def plot_progress(self):
        """
        Plot loss of NN for every image it was trained
        """
        
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker=".", grid=True, 
                yticks=(0, 0.25, 0.5, 0.1, 5.0), title="Generator Loss")
