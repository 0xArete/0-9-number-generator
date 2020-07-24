### 0-9-number-generator

# **CGAN (Convolutional generative adversarial network) -> Generating Numbers**
----------------------------
### **Third-party modules:**
- Pytorch
- Numpy
- Matplotlib
- Pandas
- tqdm
----------------------------
![GUI - gif](https://user-images.githubusercontent.com/57571014/88428244-d48dd280-cdf4-11ea-88bc-e1a913280540.gif)

## **How CGAN works?**

You create two neural networks -> Discriminator and Generator. 

In loop of training:
1. Discriminator is trained on real data from dataset with target **1.0**.
2. Discriminator is trained on fake data from generator with target **0.0**.
3. Generator is trained if it can outwit discriminator; if not than backpropagate with loss function from discriminator.

At the beginnig Generator output random noise, but through training it will learn to output image similiar from MNIST:
![through first epoch gif](https://user-images.githubusercontent.com/57571014/88433211-f049a680-cdfd-11ea-8ffa-7ba9d0fd2222.gif)

Because it get random seed (standard is: pytorch.randn(100)) and it is not trained yet. But everytime he can't outwit Discriminator, which is trained on real images of mnist dataset, it will backpropagate and learn how to generate images similiar to mnist dataset in order to fool Discriminator.

Instead of using linear nodes I used convolutional nodes. I got better results than with GAN (you can find code in "old_master" branch).

**GAN**
![digit 2020-07-13 18-06-25](https://user-images.githubusercontent.com/57571014/87326958-ba6e0d80-c533-11ea-9889-a7cceaf5126d.png)

**CGAN**
![digit 2020-07-24 20-52-10](https://user-images.githubusercontent.com/57571014/88428580-73b2ca00-cdf5-11ea-9ce9-0b40ede98ac0.png)

Also we can generate random seeds and deduce them or add them together. It's also in that picture (up). Pentultimate in first and second rows are *seed1* and *seed2*. The last one in first row and second row is *seed1-seed2* and *seed1+seed2*.

**Role of discriminator and generator**

Job of Discriminator is to recognize if it is real image or fake one produced by generator. If Discriminator think it is real image; it will output 1.0 else 0.0. And everytime it guessed wrong; it will backpropagate.

So Generator and Discriminator have competation, where Generator wants to generate fake data and outwit Discriminator and Discriminator try to guess, if it is from Generator or from real MNIST dataset. After that we can take generator, which learned to generate 28x28 digits like from MNIST.

*With GAN you have to be aware of mode collapse, where generator would output still one digit. From what I have found it's not clear why it happens, but the quality (of NN structure) and quantity (of training) plays big role.*


## **Note**
I already done this project, but this time I added concvolution neural network. I think it have better result. Also I used google collabs, where they offer gpu for free (so I want thank them for that). That's reason this time the code is written in ipynb file. If you want check previous code, than go to "old_master branch".
