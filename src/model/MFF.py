import torch
import torch.nn as nn
import torch.optim as optim

class RP_kernel(nn.Module):
    def __init__(self):
        super(RP_kernel, self).__init__()
        self.rp_kernel_en = nn.Sequential(
            nn.Linear(512,8),
            nn.ReLU()
        )
        self.rp_kernel_de = nn.Linear(8, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MTF_kernel(nn.Module):
    def __init__(self):
        super(RP_kernel, self).__init__()
        self.rp_kernel_en = nn.Sequential(
            nn.Linear(512,8),
            nn.ReLU()
        )
        self.rp_kernel_de = nn.Linear(8, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class GASF_kernel(nn.Module):
    def __init__(self):
        super(RP_kernel, self).__init__()
        self.rp_kernel_en = nn.Sequential(
            nn.Linear(512,8),
            nn.ReLU()
        )
        self.rp_kernel_de = nn.Linear(8, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class GADF_kernel(nn.Module):
    def __init__(self):
        super(RP_kernel, self).__init__()
        self.rp_kernel_en = nn.Sequential(
            nn.Linear(512,8),
            nn.ReLU()
        )
        self.rp_kernel_de = nn.Linear(8, 512)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x