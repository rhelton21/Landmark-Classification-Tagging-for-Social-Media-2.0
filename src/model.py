import torch
import torch.nn as nn

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()

        channels = [3,] + [2**(4+i) for i in range(7)]

        # Convolutional layers
        self.convs = nn.ModuleList()
        for i in range(7):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding="same"),
                nn.BatchNorm2d(channels[i+1]),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2)
            ))

        # Fully connected layers for Head
        self.fc1 = nn.Linear(channels[-1], channels[-2])
        self.fc2 = nn.Linear(channels[-2], num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through Convolutional layers
        for conv in self.convs:
            x = conv(x)

        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers with dropout and activations
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Example usage
# model = MyModel(num_classes=23, dropout=0.3)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
