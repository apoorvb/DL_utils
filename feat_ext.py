# code for models with pretrained models for feature extraction.
class feat_ext(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(feat_ext, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        mods = list(self.resnet.children())[:-3]#removing last three layers. layer4, avgpool, fc
        self.resnet = nn.Sequential(*mods)
        for p in self.resnet.parameters():
          p.requires_grad = False
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                        nn.Flatten(), 
                                        nn.Linear(256, 10, bias=True)) 
        # 10 classes in MNIST. original resnet had out of 1000 classes.
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)

        return x
