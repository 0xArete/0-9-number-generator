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



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Create structure of NN
        """

        # Neural Network layers
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
            )

        # Binary cross entropy loss -> better for classification than MSELoss
        self.loss_function = nn.BCELoss()

        # Adam optimiser; better for this task than Stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

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
        self.progress.append(loss.item())


    def plot_progress(self):
        """
        Plot loss of NN for every image it was trained
        """
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker=".", grid=True, yticks=(0, 0.25, 0.5))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Create structure of NN
        """

        # Neural Network layers
        self.model = nn.Sequential(
            nn.Linear(1, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200)
            nn.Linear(200, 784),
            nn.Sigmoid()
            )

        # No loss function; will use one from discriminator to calculate error

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)

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
        self.progress.append(loss.item())

    def plot_progress(self):
        """
        Plot loss of NN for every image it was trained
        """

        # x_size = [i for i in range(len(self.progress))]
        # # Plot
        # fig, ax = plt.subplots(figsize=(16, 8))
        # ax.set_title("Loss over time (x=trained images, y=loss values)")
        # ax.plot(x_size, self.progress)
        # plt.show()
        
        df = pd.DataFrame(self.progress, columns=["loss"])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker=".", grid=True, yticks=(0, 0.25, 0.5))