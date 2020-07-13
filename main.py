import torch
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from time import strftime, localtime

def generate_random_seed(i): 
    return torch.randn(i)

def load_generator():
    PATH = "models/generator"
    model = torch.load(PATH)
    return model

G = load_generator()
g_input_layer = 100

if __name__ == "__main__":
    class WindowDigitGenerator():

        def __init__(self):
            self.width = 1000
            self.height = 800

            self.root = tk.Tk()
            self.root.title("0-9 NUMBER GENERATOR by ludius0")
            self.root.geometry(f"{self.width}x{self.height}")

            text = "Generate digits"
            self.button1 = tk.Button(self.root, text=text, command=self.show_digits)
            self.button1.pack(side=tk.TOP, fill=tk.BOTH)

            text = "Generate single digit"
            self.button2 = tk.Button(self.root, text=text, command=lambda: self.show_digits(show_more_digits=False))
            self.button2.pack(side=tk.TOP, fill=tk.BOTH)

            self.button3 = tk.Button(self.root, text="Save chart", command=self.save_digits)
            self.button3.pack(side=tk.TOP, fill=tk.BOTH)

            self.first_click = True
            self.show_digits()
            self.root.mainloop()


        def show_digits(self, show_more_digits=True):
            if self.first_click == False:
                self.canvas.get_tk_widget().destroy()

            if show_more_digits == True:
                rows = 2
                columns = 4

                seed1 = generate_random_seed(g_input_layer)
                seed2 = generate_random_seed(g_input_layer)

                fig, ax = plt.subplots(rows, columns, figsize=(16, 8))
                for i in range(rows):
                    for j in range(columns):
                        if i == rows-1 and j == columns:        # Last in first row (penultimate row) is seed1-seed2
                            output = G.forward(seed1-seed2)
                        elif i == rows and j == columns:        # Last in second row (last row) is seed1+seed2
                            output = G.forward(seed1+seed2)
                        elif i == rows-1 and j == columns-1:    # Penultimate in first row (penultimate row) is seed1
                            output = G.forward(seed1)
                        elif i == rows and j == columns-1:      # Penultimate in second row (last row) is seed2
                            output = G.forward(seed2)
                        else:                                   # Everyone else is random
                            output = G.forward(generate_random_seed(g_input_layer))
                        img = output.detach().numpy().reshape(28, 28)
                        ax[i,j].imshow(img, interpolation="none", cmap="Blues")
            else:
                fig, ax = plt.subplots(figsize=(16, 8))
                output = G.forward(generate_random_seed(g_input_layer))
                img = output.detach().numpy().reshape(28, 28)
                ax.imshow(img, interpolation="none", cmap="Blues")

            self.canvas = FigureCanvasTkAgg(fig, master=self.root)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

            self.first_click = False

        def save_digits(self):
            timestamp = strftime("%Y-%m-%d %H-%M-%S", localtime())
            plt.savefig(f"digits/digit {timestamp}.png")

    app = WindowDigitGenerator()
