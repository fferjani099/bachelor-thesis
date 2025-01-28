import torch
import torch.nn as nn
from torchvision.models import resnet50

class MalfunctionResNet(nn.Module):
    """
    This model is basically ResNet-50 repeated 'repeats' times in the forward pass.
    The higher the 'repeats', the more GPU time it takes.
    """
    def __init__(self, repeats=5):
        super(MalfunctionResNet, self).__init__()
        self.core = resnet50(pretrained=True)
        self.repeats = repeats

    def forward(self, x):
        out = self.core(x)

        for i in range(self.repeats):
            tmp = self.core(x)     
            out = out + tmp        

        return out

if __name__ == "__main__":
    repeats = 10
    heavy_model = MalfunctionResNet(repeats=repeats).cuda().eval()

    dummy_input = torch.ones(1, 3, 224, 224).cuda()

    traced_heavy = torch.jit.trace(heavy_model, dummy_input)
    output_file = f"resnet50_malfunction_trace_{repeats}.pt"
    traced_heavy.save(output_file)
    print(f"Traced heavy ResNet saved to {output_file}")
