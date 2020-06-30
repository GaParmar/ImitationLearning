import torch
from torch import nn

class LinearPolicy(nn.Module):
    def __init__(self, output_ch=2):
        super().__init__()
        self.p = 0.2
        self.output_ch = output_ch
        self.bn = nn.BatchNorm2d(3)
        self.conv2d_1 = nn.Sequential(
                            nn.Conv2d(3, 24,kernel_size=5, stride=2, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=self.p))
        self.conv2d_2 = nn.Sequential(
                            nn.Conv2d(24, 32,kernel_size=5, stride=2, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=self.p))
        self.conv2d_3 = nn.Sequential(
                            nn.Conv2d(32, 64,kernel_size=5, stride=2, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=self.p))
        self.conv2d_4 = nn.Sequential(
                            nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=self.p))
        self.conv2d_5 = nn.Sequential(
                            nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=0,),
                            nn.ReLU(),
                            nn.Dropout2d(p=self.p))
        
        in_size=6656#2496
        # FC1
        self.fc1 = nn.Sequential(
            nn.Linear(in_size, 100),
            nn.ReLU(),
            nn.Dropout(p=self.p)
        )

        # FC2
        self.fc2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=self.p)
        ) 
        if self.output_ch == 2:
            self.fc_throttle = nn.Linear(50, 1)
            self.fc_steer = nn.Linear(50, 1)
        elif self.output_ch == 1:
            self.fc_steer = nn.Linear(50,1)
    
    def forward(self, img):
        batch = img.shape[0]
        x = self.bn(img)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = x.view(batch, -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        if self.output_ch == 2:
            throttle_logits = self.fc_throttle(x)
            steer_logits = self.fc_steer(x)
            return throttle_logits, steer_logits
        elif self.output_ch == 1:
            steer_logits = self.fc_steer(x)
            return steer_logits