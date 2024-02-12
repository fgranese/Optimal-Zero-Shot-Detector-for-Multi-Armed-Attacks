import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, input_shape, nodes, layers, drop=0.5):
        super(Detector, self).__init__()
        self.num_channels = 1
        self.input_shape = input_shape
        self.layers = layers
        activ = nn.ReLU(True)
        self.layer1 = nn.Linear(input_shape, nodes)
        self.relu = activ
        self.drop = nn.Dropout(drop)

        if layers == 3:
            self.layer2 = nn.Linear(nodes, nodes)
            self.layer3 = nn.Linear(nodes, 1)
        elif layers == 2:
            self.layer2 = nn.Linear(nodes, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if layers == 3:
            nn.init.constant_(self.layer3.weight, 0)
            nn.init.constant_(self.layer3.bias, 0)
        elif layers == 2:
            nn.init.constant_(self.layer2.weight, 0)
            nn.init.constant_(self.layer2.bias, 0)

    def forward(self, input):
        if self.layers == 3:
            f1 = self.relu(self.layer1(input))
            f2 = self.relu(self.layer2(f1))
            logits = self.layer3(self.drop(f2))
        elif self.layers == 2:
            f1 = self.relu(self.layer1(input))
            logits = self.layer2(self.drop(f1))
        return logits
