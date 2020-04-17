import os

import numpy as np
from matplotlib import pyplot as plt


class Graph:

    def __init__(self, path):
        self.path = str(path)
        self.array = []
        self.sub_coordinates = []

    def graph(self):
        if os.path.exists(self.path) and self.path.endswith('.log'):
            with open(self.path) as file:
                Graph.decode(self, file)
                fig, ax = plt.subplots()

                epochs, loss = map(list, zip(*self.array))
                ax.plot(epochs, loss, '-', lw=1)
                plt.xlabel("EPOCH", fontsize=16)
                plt.ylabel("LOSS", fontsize=16)

                plt.grid()
                plt.show()
        else:
            print("PATH DOES NOT EXISTS!")
            print("\n or extensions is not '.log'!")

    def decode(self, file):
        coordinates = []
        sub_coordinates = []
        count = 0
        pos = 0
        for i, row in enumerate(file):
            if i == 0:
                continue
            if row[i,0] == 0:
                ++count
                pos = i
            if count == 2:
                sub_coordinates.append([pos, i - 1])
                count == 0
            x, y = str(row).split(",")
            coordinates.append([x, str(y).replace("\n", "")])
        self.array = coordinates
        self.sub_coordinates = sub_coordinates
        del coordinates
