from torch import nn, optim

from .config import PSSM_ROWS, WINDOW_SIZE, EMBED


class NetPred(nn.Module):
    pth_name = f'netpred{"-full" if not EMBED else ""}.pth'

    def __init__(self):
        super().__init__()

        self.prediction = PredictionNetwork()
        self.relu = nn.ReLU()
        self.filtering = FilteringNetwork()

        print(self)

    def forward(self, x):
        """I'm not using a stack here so that the sub-models can be addressed in captum."""
        x = x.flatten(start_dim=1)
        x = self.prediction(x)

        # here PSIPRED does something different. Instead of the two networks being connected
        # like this, it modifies x to add back in the terminal region indicator dimension
        x = self.relu(x)

        x = self.filtering(x)
        return x

    def make_prediction(self, x) -> int:
        """Convenience method to make a prediction for analyses."""
        return self(x.reshape(1, 315)).argmax().item()


class PredictionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(PSSM_ROWS * WINDOW_SIZE, WINDOW_SIZE * 9),
            nn.ReLU(),
            nn.Linear(WINDOW_SIZE * 9, WINDOW_SIZE * 3),
        )

    def forward(self, x):
        return self.stack(x)


class FilteringNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(WINDOW_SIZE * 3, WINDOW_SIZE * 3),
            nn.ReLU(),
            nn.Linear(WINDOW_SIZE * 3, 3),
        )

    def forward(self, x):
        return self.stack(x)


net = NetPred()
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', patience=1, factor=0.5, verbose=True)
