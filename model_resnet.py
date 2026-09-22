import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Transfer(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':

    model = ResNet18Transfer(freeze=True, num_classes=3)

    # to find out how the modules are named:
    for name, module in model.backbone.named_modules():
        print(f"Name: {name} | Module: {module}")