import torch
import matplotlib.pyplot as plt
import numpy as np

import networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Prepare networks ###
path = "mnist_dataset/mnist_train.cvs"
dataset = networks.MnistDataset(path)

D = networks.Discriminator()
G = networks.Generator()

g_input_layer = G.input_size # 100


### FUNCTIONS ###
def plot_networks_outputs():
    """
    Plot loss function through time and generator outputs
    """
    D.plot_progress()
    G.plot_progress()

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(g_input_layer))
            img = output.detach().numpy().reshape(28, 28)
            ax[i,j].imshow(img, interpolation="none", cmap="Blues")
    plt.show()

def plot_through_epochs(evolution, plot_size):
    """
    Plot Generator outputs through epochs
    """
    fig, ax = plt.subplots(1, plot_size+1, figsize=(16, 8))
    for index, g_output in enumerate(evolution):
        img = g_output.numpy().reshape(28, 28)
        ax[index].imshow(img, interpolation="none", cmap="Blues")
    PATH = "digits/results after each epochs.png"
    plt.savefig(PATH)
    plt.show()

def save_model():
    PATH = "models/generator"
    torch.save(G, PATH)
    PATH = "models/discriminator"
    torch.save(D, PATH)

def generate_random_seed(i): 
    return torch.randn(i)


### TRAIN ###
seed1 = generate_random_seed(g_input_layer)
evolution_through_epochs = [G.forward(seed1).detach()]

epochs = 8
for e in range(1, epochs+1):
    print(f"Training in {e} / {epochs} epochs...")
    for dx, (label, img_data_tensor, target_tensor) in enumerate(dataset):
        # 1.0 for image from real dataset; 0.0 for fake one

        # train D on real/true data
        D.train(img_data_tensor, torch.FloatTensor([1.0]))

        # train D on fake/false data
        # detach() to not calculate gradients in G (for computational cost)
        D.train(G.forward(generate_random_seed(g_input_layer)).detach(), torch.FloatTensor([0.0]))

        # train G (generator)
        G.train(D, generate_random_seed(g_input_layer), torch.FloatTensor([1.0]))

        if (dx+1) % 20000 == 0:
            print(f"Trained on {dx+1} images")
        pass

    # Save progress
    evolution_through_epochs.append(G.forward(seed1).detach())


### PLOT AND SAVE ###
try:
    plot_through_epochs(evolution_through_epochs, epochs)
    plot_networks_outputs()
except:
    print("Couldn't print plots.\nProbably small range of training.")
save_model()

print("Training is done and model is saved.")
