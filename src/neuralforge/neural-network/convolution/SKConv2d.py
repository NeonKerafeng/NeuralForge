import torch

class SKConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        M: int = 2,
        r: int = 16,
        L: int = 32,
        groups: int = 1
    ):

        super().__init__() # type: ignore

        self.convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 3,
                    stride = stride,
                    padding = 1 + i,
                    dilation = 1 + i,
                    groups = groups,
                    bias = False
                ),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            ) for i in range(M)
        ])

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        d = max(out_channels // r, L)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(out_channels, d),
            torch.nn.BatchNorm1d(d),
            torch.nn.ReLU(inplace=True)
        )

        self.fcs = torch.nn.ModuleList([
            torch.nn.Linear(d, out_channels) for _ in range(M)
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        threads = [f(x) for f in self.convs] # List of [B, out_channels, ...shape]
        feats = torch.stack(threads, dim=1) # [B, M, out_channels, ...shape]

        U = feats.sum(dim=1) # [B, M, out_channels, ...shape] -> [B, out_channels, ...shape]
        Z = self.fc(self.gap(U).view(batch_size, -1)) # [B, out_channels, ...shape] -> [B, out_channels, 1, 1] -> [B, out_channels] -> [B, d]

        weights = torch.stack([fc(Z) for fc in self.fcs], dim=1) # [B, d] -> [B, out_channels] -> [B, M, out_channels]
        weights = self.softmax(weights).unsqueeze(-1).unsqueeze(-1) # [B, M, out_channels, 1, 1]

        res = (feats * weights).sum(dim=1) # [B, M, out_channels, ...shape] -> [B, out_channels, ...shape]

        return res

class SKUnit(torch.nn.Module):
    is_preact = True

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        stride: int = 1,
        M: int = 2,
        r: int = 16,
        L: int = 32,
        groups: int = 1,
        mode: str = "v2"
    ):
        super().__init__() # type: ignore

        if mode not in ["v1", "v2"]:
            raise NotImplementedError(f"Current mode '{mode}' is not available for SKUnit.")

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if mode == 'v1':
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

        if mode == 'v1':
            self.is_preact = False

            self.project = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, intermediate_channels, 1, bias=False),
                torch.nn.BatchNorm2d(intermediate_channels),
                torch.nn.ReLU(inplace=True)
            )

            self.transform = torch.nn.Sequential(
                SKConv2d(
                    intermediate_channels,
                    intermediate_channels,
                    stride = stride,
                    M = M,
                    r = r,
                    L = L,
                    groups = groups
                ),
                torch.nn.BatchNorm2d(intermediate_channels),
                torch.nn.ReLU(inplace=True)
            )

            self.lift = torch.nn.Sequential(
                torch.nn.Conv2d(intermediate_channels, out_channels, 1, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

        else:
            self.project = torch.nn.Sequential(
                torch.nn.BatchNorm2d(in_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels, intermediate_channels, 1, bias=False)
            )

            self.transform = torch.nn.Sequential(
                torch.nn.BatchNorm2d(intermediate_channels),
                torch.nn.ReLU(inplace=True),
                SKConv2d(
                    intermediate_channels,
                    intermediate_channels,
                    stride = stride,
                    M = M,
                    r = r,
                    L = L,
                    groups = groups
                )
            )

            self.lift = torch.nn.Sequential(
                torch.nn.BatchNorm2d(intermediate_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(intermediate_channels, out_channels, 1, bias=False)
            )

        self.final_ReLU = torch.nn.ReLU(inplace=True)
        self.mode = mode

    def forward(self, x: torch.Tensor):
        residual = self.shortcut(x)

        x = self.project(x)
        x = self.transform(x)
        x = self.lift(x)

        return self.final_ReLU(x + residual) if self.mode == 'v1' else x + residual
