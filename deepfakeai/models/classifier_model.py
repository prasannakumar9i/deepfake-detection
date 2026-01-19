import torch.nn as nn

class ClassifierHead(nn.Module):
    """
    Final Classifier
    """
    def __init__(
        self, 
        input_dim, 
        num_classes=2
    ):
        super().__init__()
        self.fc = nn.Linear(
            input_dim, 
            num_classes
        )

    def forward(self, x):
        return self.fc(x)
