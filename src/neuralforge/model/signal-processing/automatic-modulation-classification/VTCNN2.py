import torch

class VTCNN2(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__() # type: ignore

        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3)),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5)
            ),

            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3)),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5)
            ),

            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.LazyLinear(out_features=256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5)
            ),

            torch.nn.Sequential(
                torch.nn.LazyLinear(out_features=num_classes)
            )
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
