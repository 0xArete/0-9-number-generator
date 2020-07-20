import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class MnistDataset(Dataset):
    def __init__(self, path):
        """
        Load data in class
        """
        # CSV reader
        training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
        self.data_df = training_data_file.readlines()
        training_data_file.close()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        """
        Get from 'for loop' (label, tensor raw image, target of image (what should NN output))
        """
        img = list(map(int, self.data_df[index].split(',')))
        label = img[0]
        target = torch.zeros((10))
        target[label] = 1.0
        img_values = torch.FloatTensor(img[1:]) / 255.0

        return label, img_values, target

    def plot(self, index):
        """
        Plot (show) targeted (index) image
        """
        img = list(map(int, self.data_df[index].split(',')))
        arr = np.array(img[1:]).reshape(28, 28)
        plt.title("label = " + str(img[0]))
        plt.imshow(arr, interpolation="none", cmap="Blues")
        plt.show()


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        """Reshape to another dimensnsion"""
        self.shape = shape, # <- create tuple
    
    def forward(self, x):
        return x.reshape(*self.shape)



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Create structure of NN
        """

        # Neural Network layers
        self.model = nn.Sequential(
            View((1, 1, 28, 28)),
            nn.Conv2d(1, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),

            View(1)
            )

        # Binary cross entropy loss -> better for classification than MSELoss
        self.loss_function = nn.BCELoss()

        # Adam optimiser; better for this task than Stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Timestap of loss for ploting progress
        self.progress = []
        self.counter = 0

    def forward(self, inputs):
        """
        Pass through NN and get its answer
        """
        return self.model(inputs)

    def train(self, inputs, targets):
        """
        Train NN; Take tensor of image with label identificator of image;
        Pass through NN; get loss/cost function and backpropagate NN
        to tweak weights (layers)
        """
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        # Backpropagation
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Timestamp
        self.counter += 1
        if self.counter % 100 == 0:
            self.progress.append(loss.item())


    def plot_progress(self):
        """
        Plot loss of NN for every image it was trained
        """
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker=".", grid=True,
                yticks=(0, 0.25, 0.5, 0.1, 5.0), title="Discriminator Loss")


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
