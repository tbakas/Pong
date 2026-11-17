from torch.nn import Linear, Sequential, LeakyReLU, Module
from torch import save, load
from torch.optim import Adam, AdamW

input_size = 6
actor_hidden_size = 100
actor_blocks = 3


class Model(Module):
    def __init__(self):
        super().__init__()
        layers = [Linear(5, actor_hidden_size), LeakyReLU()]
        for i in range(actor_blocks):
            layers.extend([Linear(actor_hidden_size, actor_hidden_size), LeakyReLU()])
        layers.append(Linear(actor_hidden_size, 3))
        self.sequential = Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class Critic(Module):
    def __init__(self):
        super().__init__()
        layers = [Linear(5, actor_hidden_size), LeakyReLU()]
        for i in range(actor_blocks):
            layers.extend([Linear(actor_hidden_size, actor_hidden_size), LeakyReLU()])
        layers.append(Linear(actor_hidden_size, 1))
        self.sequential = Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)
