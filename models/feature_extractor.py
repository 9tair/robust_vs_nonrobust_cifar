import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])  # Remove last classification layer

    def forward(self, x):
        x = self.features(x)
        return x.view(x.shape[0], -1)  # Flatten to feature vector
