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


class TrackModel(Module):
    def __init__(self):
        super().__init__()
        layers = [Linear(6, actor_hidden_size), LeakyReLU()]
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


if __name__ == '__main__':
    model = Critic()
    print('critic:', sum(p.numel() for p in model.parameters()), 'parameters')
    save(model.state_dict(), "critic.pth")
    optim = AdamW(model.parameters(), lr=1e-4)
    save(optim.state_dict(), 'critic_optimizer.pth')

    model = Model()
    print('actor:', sum(p.numel() for p in model.parameters()), 'parameters')

    optim = Adam(model.parameters(), lr=3e-4)
    save(optim.state_dict(), 'model_optimizer.pth')

    save(model.state_dict(), "actor.pth")
