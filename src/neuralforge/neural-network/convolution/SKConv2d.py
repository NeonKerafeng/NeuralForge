import torch

class SKConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32, G=1):
        super().__init__()
        self.M = M

        self.convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                                padding=1+i, dilation=1+i, bias=False, groups=G),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
            )
            for i in range(M)
        ])

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        d = max(int(out_channels / r), L)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(out_channels, d),
            torch.nn.BatchNorm1d(d),
            torch.nn.ReLU(inplace=True)
        )

        self.fcs = torch.nn.ModuleList([
            torch.nn.Linear(d, out_channels) for _ in range(M)
        ])

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)

        feats = [f(x) for f in self.convs]

        feats_tensor = torch.stack(feats, dim=1)

        U = feats_tensor.sum(dim=1)
        Aux = self.gap(U).view(batch_size, -1)
        Z = self.fc(Aux)

        weights = torch.stack([fc(Z) for fc in self.fcs], dim=1)
        weights = self.softmax(weights)

        weights = weights.unsqueeze(-1).unsqueeze(-1) 

        res = (feats_tensor * weights).sum(dim=1)

        return res

class SKUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32, G=1):
        super().__init__()
        self.conv = SKConv2d(in_channels, out_channels, stride, M, r, L, G)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        res = self.bn(self.conv(x))
        res += self.shortcut(x)
        return self.relu(res)
