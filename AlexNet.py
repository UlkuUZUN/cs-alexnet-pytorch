import torch.nn as nn
import torch
import torch.nn.functional as F
'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.conv=nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        all_shifts = [(0, 0),
        
                      (1, 0, 0, 0), (0, 0, 1, 0),
                      (1, 0, 1, 0), (1, 0, 0, 1), 
                      (0, 1, 0, 1), (0, 1, 1, 0),
                      (0, 1, 0, 0), (0, 0, 0, 1),
                      
                      (2, 0, 0, 0), (0, 0, 2, 0),
                      (2, 0, 2, 0), (2, 0, 0, 2), 
                      (0, 2, 0, 2), (0, 2, 2, 0),
                      (0, 2, 0, 0), (0, 0, 0, 2),
        
        
                     ]
        outputs = []
        output0 = self.conv(x)
        outputs.append(output0)
        (batch, channel, h, w) = output0.size()
        
        for shift in all_shifts[1:]:
            padded = F.pad(x, (shift[0],shift[1],shift[2],shift[3]), mode='constant')
            cropped=padded[:, :, 0 : 32, 0 : 32 ]
            output = self.conv(cropped)
            output = output[:, :, 0 : 16, 0 : 16 ]
            #output = output[:, :, 0 + shift[2]: h - shift[3], 0 + shift[0]: w - shift[1]]
            
            outputs.append(output)
        median, input_indexes =  torch.median(torch.stack(outputs), 0)
        x=median
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
