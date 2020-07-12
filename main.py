import torch
from numpy import random
import matplotlib.pyplot as plt

import networks

path = "mnist_dataset/mnist_train.cvs"
dataset = networks.MnistDataset(path)
D = networks.Discriminator()
G = networks.Generator()

def generate_random(i): 
    """
    Generate random digit we want as representation
    """
    # We could pick random one from 0 to 9, but we can always generate new "same" digit
    return torch.rand(i)

### TRAIN ###
epochs = 1
for e in range(1, epochs+1):
    print(f"Training in {e} / {epochs} epochs...")
    for dx, (label, img_data_tensor, target_tensor) in enumerate(dataset):
        # 1.0 for image from real dataset; 0.0 for fake one

        # train D on real/true data
        D.train(img_data_tensor, torch.FloatTensor([1.0]))

        # train D on fake/false data
        # detach() to not calculate gradients in G (for computational cost)
        D.train(G.forward(generate_random(1)).detach(), torch.FloatTensor([0.0]))

        # train G (generator)
        G.train(D, generate_random(1), torch.FloatTensor([1.0]))

        if (dx+1) % 20000 == 0:
            print(f"Trained on {dx+1} images")



D.plot_progress()
G.plot_progress()


fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for i in range(2):
    for j in range(3):
        output = G.forward(generate_random(1))
        img = output.detach().numpy().reshape(28, 28)
        ax[i,j].imshow(img, interpolation="none", cmap="Blues")
plt.show()
