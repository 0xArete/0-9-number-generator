# 0-9-number-generator

**GAN (Generative adversarial network) -> Generating Numbers**
----------------------------
##**Third-party modules:**
- Pytorch
- Numpy
- Matplotlib
- Pandas

![ezgif com-gif-maker2](https://user-images.githubusercontent.com/57571014/87329620-85fc5080-c537-11ea-9118-cef122809059.gif)

**How GAN works?**

You create two neural networks -> Discriminator and Generator. 

In loop of training:
1. Discriminator is trained on real data from dataset with target **1.0**.
2. Discriminator is trained on fake data from generator with target **0.0**.
3. Generator is trained if it can outwit discriminator; if not than backpropagate with loss function from discriminator.

At the beginnig Generator output something like this:
![withou training](https://user-images.githubusercontent.com/57571014/87323939-b6d88780-c52f-11ea-9b77-daa07d1211f6.png)

Because it get random seed (in my case: pytorch.randn(100)) and it is not trained yet. But everytime he can't outwit Discriminator, which is trained on real images of mnist dataset, it will backpropagate and learn how to generate images similiar to mnist dataset.

Job of Discriminator is to recognize if it is real image or fake one produced by generator. If Discriminator think it is real image; it will output 1.0 else 0.0. And everytime it guessed wrong; it will backpropagate.

So Generator and Discriminator have competation, where G wants to generate fake data and outwit Discriminator and Discriminator try to guess, if it is from Generator or from real MNIST dataset. After that we can take generator, which learned to generate 28x28 digits like from MNIST.

In my case after 8 epochs on 60000 training MNIST dataset I git something like this:
![digit 2020-07-13 18-06-25](https://user-images.githubusercontent.com/57571014/87326958-ba6e0d80-c533-11ea-9889-a7cceaf5126d.png)

Also we can generate random seeds and deduce them or add them together. It's also in that picture (up). Pentultimate in first and second rows are seed1 and seed2 (pytorch.randn(100)). The last one in first row and second row is seed1-seed2 and seed1+seed2.

With GAN you have to be aware of mode collapse, where generator would output still one digit. From what I have found it's not clear why it happens, but the quality (of NN structure) and quantity (of training) plays big role.


**Network Structure**\
For Discriminator I used 784 input layer (mnist 28x28=784 image) to 200; LeakyReLU(0.02); Layer Normalisation of 200 (cram in to 0 to 1), Sigmoid (200) to 1 -> because we want only one output 1.0 (it is from MNIST) or 0.0 (it is from Generator).

For Generator I mirrored it; except input layer -> It is random seed of 100 (pytorch.randn(100)).

As optimiser I used Adam for both networks (gives that "rolling ball" on gradient descent velocity) and as Loss function I used BCELoss, because it is better for classification (I gave it only to Discriminator; Generator use one from Discriminator)


**Note**\
I am suprised how well this project turned out; I wasn't expecting any results like that. I did use framework, but this time I knew what I was doing (thanks to my previous project Number guesser, which I did without framework) and later it would be benefiting use one (Mainly pytorch; it's not as easy for beginners like Keras, where beginner can make NN and still not understand what he has done and pytorch is lighweight framework and pythonic; so it is good choice). There aren't many great resources on neural networks or on pytorch, but I can recommend Tariq Rashid or Sentdex.
